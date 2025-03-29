// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{Array, ArrayAccessor, ArrayBuilder, ArrayRef, AsArray, Datum, Int64Array, StringArray, StringArrayType, GenericStringArray};
use arrow::datatypes::{DataType, Int64Type, StringViewType, ByteViewType};
use arrow::datatypes::{
    DataType::Int64, DataType::LargeUtf8, DataType::Utf8, DataType::Utf8View,
};
use arrow::error::ArrowError;
use datafusion_common::{exec_err, internal_err, Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarUDFImpl, Signature, TypeSignature::Exact,
    TypeSignature::Uniform, Volatility,
};
use datafusion_macros::user_doc;
use itertools::izip;
use regex::Regex;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;

#[user_doc(
    doc_section(label = "Regular Expression Functions"),
    description = "Returns the position in a string where the specified occurrence of a POSIX regular expression is located.",
    syntax_example = "regexp_instr(str, regexp[, start, N, endoption, flags])",
    sql_example = r#"```sql
> SELECT regexp_instr('ABCDEF', 'c(.)(..)', 1, 1, 0, 'i', 2);
+---------------------------------------------------------------+
| regexp_instr()                                                |
+---------------------------------------------------------------+
| 5                                                             |
+---------------------------------------------------------------+
```"#,
    standard_argument(name = "str", prefix = "String"),
    standard_argument(name = "regexp", prefix = "Regular"),
    argument(
        name = "start",
        description = "- **start**: Optional start position (the first position is 1) to search for the regular expression. Can be a constant, column, or function. Defaults to 1"
    ),
    argument(
        name = "N",
        description = "- **N**: Optional The N-th occurrence of pattern to find. Defaults to 1 (first match). Can be a constant, column, or function."
    ),
    argument(
        name = "endoption",
        description = "- **endoption**: Optional. If 0, returns the starting position of the match (default). If 1, returns the ending position of the match. Can be a constant, column, or function."
    ),
    argument(
        name = "flags",
        description = r#"Optional regular expression flags that control the behavior of the regular expression. The following flags are supported:
  - **i**: case-insensitive: letters match both upper and lower case
  - **m**: multi-line mode: ^ and $ match begin/end of line
  - **s**: allow . to match \n
  - **R**: enables CRLF mode: when multi-line mode is enabled, \r\n is used
  - **U**: swap the meaning of x* and x*?"#
    ),
    argument(
        name = "subexpr",
        description = "Optional Specifies which capture group (subexpression) to return the position for. Defaults to 0, which returns the position of the entire match."
    )
)]
#[derive(Debug)]
pub struct RegexpInstrFunc {
    signature: Signature,
}

impl Default for RegexpInstrFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl RegexpInstrFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    Uniform(2, vec![Utf8View, LargeUtf8, Utf8]),
                    Exact(vec![Utf8View, Utf8View, Int64]),
                    Exact(vec![LargeUtf8, LargeUtf8, Int64]),
                    Exact(vec![Utf8, Utf8, Int64]),
                    Exact(vec![Utf8View, Utf8View, Int64, Int64]),
                    Exact(vec![LargeUtf8, LargeUtf8, Int64, Int64]),
                    Exact(vec![Utf8, Utf8, Int64, Int64]),
                    Exact(vec![Utf8View, Utf8View, Int64, Int64, Int64]),
                    Exact(vec![LargeUtf8, LargeUtf8, Int64, Int64, Int64]),
                    Exact(vec![Utf8, Utf8, Int64, Int64, Int64]),
                    Exact(vec![Utf8View, Utf8View, Int64, Utf8View]),
                    Exact(vec![LargeUtf8, LargeUtf8, Int64, LargeUtf8]),
                    Exact(vec![Utf8, Utf8, Int64, Utf8]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for RegexpInstrFunc {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "regexp_instr"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(Int64)
    }

    fn invoke_with_args(
        &self,
        args: datafusion_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let args = &args.args;

        let len = args
            .iter()
            .fold(Option::<usize>::None, |acc, arg| match arg {
                ColumnarValue::Scalar(_) => acc,
                ColumnarValue::Array(a) => Some(a.len()),
            });

        let is_scalar = len.is_none();
        let inferred_length = len.unwrap_or(1);
        let args = args
            .iter()
            .map(|arg| arg.to_array(inferred_length))
            .collect::<Result<Vec<_>>>()?;

        let result = regexp_instr_func(&args);
        if is_scalar {
            // If all inputs are scalar, keeps output as scalar
            let result = result.and_then(|arr| ScalarValue::try_from_array(&arr, 0));
            result.map(ColumnarValue::Scalar)
        } else {
            result.map(ColumnarValue::Array)
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

pub fn regexp_instr_func(args: &[ArrayRef]) -> Result<ArrayRef> {
    let args_len = args.len();
    if !(2..=4).contains(&args_len) {
        return exec_err!("regexp_instr was called with {args_len} arguments. It requires at least 2 and at most 4.");
    }

    let values = &args[0];
    match values.data_type() {
        Utf8 | LargeUtf8 | Utf8View => (),
        other => {
            return internal_err!(
                "Unsupported data type {other:?} for function regexp_instr"
            );
        }
    }

    regexp_instr(
        values,
        &args[1],
        if args_len > 2 { Some(&args[2]) } else { None },
        if args_len > 3 { Some(&args[3]) } else { None },
        if args_len > 4 { Some(&args[4]) } else { None },
        if args_len > 5 { Some(&args[5]) } else { None },
        if args_len > 6 { Some(&args[6]) } else { None },
    )
    .map_err(|e| e.into())
}

/// `arrow-rs` style implementation of `regexp_instr` function.
/// This function `regexp_instr` is responsible for counting the occurrences of a regular expression pattern
/// within a string array. It supports optional start positions and flags for case insensitivity.
///
/// The function accepts a variable number of arguments:
/// - `values`: The array of strings to search within.
/// - `regex_array`: The array of regular expression patterns to search for.
/// - `start_array` (optional): The array of start positions for the search.
/// - `nth_array` (optional): The array of start nth for the search.
/// - `endoption_array` (optional): The array of endoption positions for the search.
/// - `flags_array` (optional): The array of flags to modify the search behavior (e.g., case insensitivity).
/// - `subexpr_array` (optional): The array of subexpr positions for the search.
///
/// The function handles different combinations of scalar and array inputs for the regex patterns, start positions,
/// and flags. It uses a cache to store compiled regular expressions for efficiency.
///
/// # Errors
/// Returns an error if the input arrays have mismatched lengths or if the regular expression fails to compile.
pub fn regexp_instr(
    values: &dyn Array,
    regex_array: &dyn Datum,
    start_array: Option<&dyn Datum>,
    nth_array: Option<&dyn Datum>,
    endoption_array: Option<&dyn Datum>,
    flags_array: Option<&dyn Datum>,
    subexpr_array: Option<&dyn Datum>,
) -> Result<ArrayRef, ArrowError> {
    let (regex_array, is_regex_scalar) = regex_array.get();
    let (start_array, is_start_scalar) = start_array.map_or((None, true), |start| {
        let (start, is_start_scalar) = start.get();
        (Some(start), is_start_scalar)
    });
    let (nth_array, is_nth_scalar) = nth_array.map_or((None, true), |nth| {
        let (nth, is_nth_scalar) = nth.get();
        (Some(nth), is_nth_scalar)
    });
    let (endoption_array, is_endoption_scalar) = endoption_array.map_or((None, true), |endoption| {
        let (endoption, is_endoption_scalar) = endoption.get();
        (Some(endoption), is_endoption_scalar)
    });
    let (flags_array, is_flags_scalar) = flags_array.map_or((None, true), |flags| {
        let (flags, is_flags_scalar) = flags.get();
        (Some(flags), is_flags_scalar)
    });
    let (subexpr_array, is_subexpr_scalar) = subexpr_array.map_or((None, true), |subexpr| {
        let (subexpr, is_subexpr_scalar) = subexpr.get();
        (Some(subexpr), is_subexpr_scalar)
    });


    match (values.data_type(), regex_array.data_type(), flags_array) {
        (Utf8, Utf8, None) => regexp_instr_inner(
            values.as_string::<i32>(),
            regex_array.as_string::<i32>(),
            is_regex_scalar,
            start_array.map(|start| start.as_primitive::<Int64Type>()),
            is_start_scalar,
            nth_array.map(|nth| nth.as_primitive::<Int64Type>()),
            is_nth_scalar,
            endoption_array.map(|endoption| endoption.as_primitive::<Int64Type>()),
            is_endoption_scalar,
            None,
            is_flags_scalar,
            subexpr_array.map(|subexpr| subexpr.as_primitive::<Int64Type>()),
            is_subexpr_scalar,
        ),
        (Utf8, Utf8, Some(flags_array)) if *flags_array.data_type() == Utf8 => regexp_instr_inner(
            values.as_string::<i32>(),
            regex_array.as_string::<i32>(),
            is_regex_scalar,
            start_array.map(|start| start.as_primitive::<Int64Type>()),
            is_start_scalar,
            nth_array.map(|nth| nth.as_primitive::<Int64Type>()),
            is_nth_scalar,
            endoption_array.map(|endoption| endoption.as_primitive::<Int64Type>()),
            is_endoption_scalar,
            Some(flags_array.as_string::<i32>()),
            is_flags_scalar,
            subexpr_array.map(|subexpr| subexpr.as_primitive::<Int64Type>()),
            is_subexpr_scalar,
        ),
        (LargeUtf8, LargeUtf8, None) => regexp_instr_inner(
            values.as_string::<i64>(),
            regex_array.as_string::<i64>(),
            is_regex_scalar,
            start_array.map(|start| start.as_primitive::<Int64Type>()),
            is_start_scalar,
            nth_array.map(|nth| nth.as_primitive::<Int64Type>()),
            is_nth_scalar,
            endoption_array.map(|endoption| endoption.as_primitive::<Int64Type>()),
            is_endoption_scalar,
            None,
            is_flags_scalar,
            subexpr_array.map(|subexpr| subexpr.as_primitive::<Int64Type>()),
            is_subexpr_scalar,
        ),
        (LargeUtf8, LargeUtf8, Some(flags_array)) if *flags_array.data_type() == LargeUtf8 => regexp_instr_inner(
            values.as_string::<i64>(),
            regex_array.as_string::<i64>(),
            is_regex_scalar,
            start_array.map(|start| start.as_primitive::<Int64Type>()),
            is_start_scalar,
            nth_array.map(|nth| nth.as_primitive::<Int64Type>()),
            is_nth_scalar,
            endoption_array.map(|endoption| endoption.as_primitive::<Int64Type>()),
            is_endoption_scalar,
            Some(flags_array.as_string::<i64>()),
            is_flags_scalar,
            subexpr_array.map(|subexpr| subexpr.as_primitive::<Int64Type>()),
            is_subexpr_scalar,
        ),
        (Utf8View, Utf8View, None) => regexp_instr_inner(
            values.as_string_view(),
            regex_array.as_string_view(),
            is_regex_scalar,
            start_array.map(|start| start.as_primitive::<Int64Type>()),
            is_start_scalar,
            nth_array.map(|nth| nth.as_primitive::<Int64Type>()),
            is_nth_scalar,
            endoption_array.map(|endoption| endoption.as_primitive::<Int64Type>()),
            is_endoption_scalar,
            None,
            is_flags_scalar,
            subexpr_array.map(|subexpr| subexpr.as_primitive::<Int64Type>()),
            is_subexpr_scalar,
        ),
        (Utf8View, Utf8View, Some(flags_array)) if *flags_array.data_type() == Utf8View => regexp_instr_inner(
            values.as_string_view(),
            regex_array.as_string_view(),
            is_regex_scalar,
            start_array.map(|start| start.as_primitive::<Int64Type>()),
            is_start_scalar,
            nth_array.map(|nth| nth.as_primitive::<Int64Type>()),
            is_nth_scalar,
            endoption_array.map(|endoption| endoption.as_primitive::<Int64Type>()),
            is_endoption_scalar,
            Some(flags_array.as_string_view()),
            is_flags_scalar,
            subexpr_array.map(|subexpr| subexpr.as_primitive::<Int64Type>()),
            is_subexpr_scalar,
        ),
        _ => Err(ArrowError::ComputeError(
            "regexp_count() expected the input arrays to be of type Utf8, LargeUtf8, or Utf8View and the data types of the values, regex_array, and flags_array to match".to_string(),
        )),
    }
    }



// pub fn cast_to_array(arg_array: Option<&Int64Array>, is_scalar: bool, cast_length: usize)  -> Result<Option<&Int64Array>, ArrowError> {
//     let arg_array = 
//     if let Some(arg_array) = arg_array {
//         if is_scalar || arg_array.len() == 1 {
//             Ok(Some((&Int64Array::from(vec![arg_array.value(0); cast_length]))))
//         } else if arg_array.len() == cast_length as usize {
//             Ok(Some(arg_array))
//         } else {
//             Err(ArrowError::ComputeError(format!(
//                 "Argument array must be the same length as values array; got {} and {}",
//                 arg_array.len(),
//                 cast_length,
//             )))
//         }
//     } else {
//             Ok(Some(&Int64Array::from(vec![1; cast_length])))
//         };
//     arg_array
//     }


// pub struct ProcessArgument<'a, T> {
//     arg_array: Option<&'a T>,
//     is_scalar: bool,
//     cast_length: usize,
// }
// pub trait GetArray {
//     fn cast_to_array<T>(&self) -> T;
// }

// impl GetArray for ProcessArgument {
//     fn cast_to_array<T>(&self) -> T {
//         let arg_array = 
//     if let Some(arg_array) = self.arg_array {
//         if self.is_scalar || arg_array.len() == 1 {
//             Ok(Some((&Int64Array::from(vec![self.arg_array.value(0); self.cast_length]))))
//         } else if self.arg_array.len() == self.cast_length as usize {
//             Ok(Some(self.arg_array))
//         } else {
//             Err(ArrowError::ComputeError(format!(
//                 "Argument array must be the same length as values array; got {} and {}",
//                 self.arg_array.len(),
//                 self.cast_length,
//             )))
//         }
//     } else {
//             Ok(Some(&Int64Array::from(vec![1; cast_length])))
//         };
//     self.arg_array
//     }
// }

enum ScalarOrArray<T> {
    Scalar(T),
    Array(Vec<T>),
}

impl< T: Clone> ScalarOrArray<T> {
    fn iter(&self, len: usize) -> Box<dyn Iterator<Item = T> + '_> {
        match self {
            ScalarOrArray::Scalar(val) => Box::new(std::iter::repeat(val.clone()).take(len)),
            ScalarOrArray::Array(arr) => Box::new(arr.iter().cloned()),
        }
    }
}

pub fn regexp_instr_inner<'a, S>(
    values: S,
    regex_array: S,
    is_regex_scalar: bool,
    start_array: Option<&Int64Array>,
    is_start_scalar: bool,
    nth_array: Option<&Int64Array>,
    is_nth_scalar: bool,
    endoption_array: Option<&Int64Array>,
    is_endoption_scalar: bool,
    flags_array: Option<S>,
    is_flags_scalar: bool,
    subexp_array: Option<&Int64Array>,
    is_subexp_scalar: bool,
) -> Result<ArrayRef, ArrowError>
where
    S: StringArrayType<'a>,
{
    let len = values.len();

    let regex_input = if is_regex_scalar || regex_array.len() == 1 {
        ScalarOrArray::Scalar(regex_array.value(0))
    } else {
        let regex_vec: Vec<&str> = regex_array.iter().map(|v| v.unwrap_or("")).collect();
        ScalarOrArray::Array(regex_vec)
    };

    let start_input = if let Some(start) = start_array {
        if is_start_scalar || start.len() == 1 {
            ScalarOrArray::Scalar(start.value(0))
        } else {
            let start_vec: Vec<i64> = start_array
                .iter()
                .map(|array| array.value(0)) // Extract the value from the array
                .collect();
            ScalarOrArray::Array(start_vec)
        }
    } else {
        
        ScalarOrArray::Scalar(1) // Default start = 1
    };

    let nth_input = if let Some(nth) = nth_array {
        if is_nth_scalar || nth.len() == 1 {
            ScalarOrArray::Scalar(nth.value(0))
        } else {
            let nth_vec: Vec<i64> = nth_array
                .iter()
                .map(|array| array.value(0)) // Extract the value from the array
                .collect();
            ScalarOrArray::Array(nth_vec)
        }
    } else {
        ScalarOrArray::Scalar(1) // Default nth = 0
    };

    let endoption_input = if let Some(endoption) = endoption_array {
        if is_endoption_scalar || endoption.len() == 1 {
            ScalarOrArray::Scalar(endoption.value(0))
        } else {
            let endoption_vec: Vec<i64> = endoption_array
                .iter()
                .map(|array| array.value(0)) // Extract the value from the array
                .collect();
            ScalarOrArray::Array(endoption_vec)
        }
    } else {
        ScalarOrArray::Scalar(0) // Default endoption = 0
    };

    let flags_input = if let Some(ref flags) = flags_array {
        if is_flags_scalar || flags.len() == 1 {
            ScalarOrArray::Scalar(flags.value(0))
        } else {
            let flags_vec: Vec<&str> = flags.iter().map(|v| v.unwrap_or("")).collect();
            ScalarOrArray::Array(flags_vec)
        }
    } else {
        ScalarOrArray::Scalar("") // Default flags = ""
    };

    let subexp_input = if let Some(subexp) = subexp_array {
        if is_subexp_scalar || subexp.len() == 1 {
            ScalarOrArray::Scalar(subexp.value(0))
        } else {
            let subexp_vec: Vec<i64> = subexp_array
                .iter()
                .map(|array| array.value(0)) // Extract the value from the array
                .collect();
            ScalarOrArray::Array(subexp_vec)
        }
    } else {
        ScalarOrArray::Scalar(0) // Default subexp = 0
    };

    let mut regex_cache = HashMap::new();

    let result: Result<Vec<i64>, ArrowError> = izip!(
        values.iter(),
        regex_input.iter(len),
        start_input.iter(len),
        nth_input.iter(len),
        endoption_input.iter(len),
        flags_input.iter(len),
        subexp_input.iter(len)
    )
    .map(|(value, regex, start, nth, endoption, flags, subexp)| {
        if regex.is_empty() {
            return Ok(0);
        }

        let pattern = compile_and_cache_regex(&regex, Some(flags), &mut regex_cache)?;

        get_index(value, &pattern, start, nth, endoption, subexp)
    })
    .collect();

    Ok(Arc::new(Int64Array::from(result?)))
}   

fn compile_and_cache_regex<'strings, 'cache>(
    regex: &'strings str,
    flags: Option<&'strings str>,
    regex_cache: &'cache mut HashMap<(&'strings str, Option<&'strings str>), Regex>,
) -> Result<&'cache Regex, ArrowError>
where
    'strings: 'cache,
{
    let result = match regex_cache.entry((regex, flags)) {
        Entry::Occupied(occupied_entry) => occupied_entry.into_mut(),
        Entry::Vacant(vacant_entry) => {
            let compiled = compile_regex(regex, flags)?;
            vacant_entry.insert(compiled)
        }
    };
    Ok(result)
}

fn compile_regex(regex: &str, flags: Option<&str>) -> Result<Regex, ArrowError> {
    let pattern = match flags {
        None | Some("") => regex.to_string(),
        Some(flags) => {
            if flags.contains("g") {
                return Err(ArrowError::ComputeError(
                    "regexp_count() does not support global flag".to_string(),
                ));
            }
            format!("(?{}){}", flags, regex)
        }
    };

    Regex::new(&pattern).map_err(|_| {
        ArrowError::ComputeError(format!(
            "Regular expression did not compile: {}",
            pattern
        ))
    })
}

fn get_index(
    value: Option<&str>,
    pattern: &Regex,
    start: i64,
    n: i64,
    endoption: i64,
    subexpr: i64,
) -> Result<i64, ArrowError> {
    let value = match value {
        None | Some("") => return Ok(0),
        Some(value) => value,
    };

    // let start = start.unwrap_or(1);
    if start < 1 {
        return Err(ArrowError::ComputeError(
            "regexp_instr() requires start to be 1-based".to_string(),
        ));
    }
    
    // let n = n.unwrap_or(1); // Default to finding the first match
    if n < 1 {
        return Err(ArrowError::ComputeError(
            "N must be 1 or greater".to_string(),
        ));
    }
    
    let find_slice = value.chars().skip(start as usize - 1).collect::<String>();
    let mut matches = pattern.find_iter(&find_slice).enumerate();
    
    // Find the N-th match (1-based index)
    let nth_match = matches
    .nth((n - 1) as usize)
    .ok_or_else(|| ArrowError::ComputeError("N-th match not found".to_string()))?;
    
    let match_start = nth_match.1.start() as i64 + start - 1;
    let match_end = nth_match.1.end() as i64 + start - 1;
    
    let result_index = match endoption {
        1 => match_end,  // Return end position of match
        _ => match_start,      // Default: Return start position
    };
    
    // Handle subexpression capturing (if requested)
    
    if subexpr > 0 {
        if let Some(captures) = pattern.captures(&find_slice) {
            if let Some(matched) = captures.get(subexpr as usize) {
                return Ok(matched.start() as i64 + start - 1);
            }
        }
        return Ok(0); // Return 0 if the subexpression was not found
    }
    
    
    Ok(result_index)
}

// pub fn get_args_tuple(arg_array: Option<&Int64Array>, is_arg_scalar: bool) -> (Option<&Int64Array>, Option<i64>, bool) {
//     let (res_array, res_scalar, is_res_scalar) =
//         if let Some(arg_array) = arg_array {
//             if is_arg_scalar || arg_array.len() == 1 {
//                 (None, Some(arg_array.value(0)), true)
//             } else {
//                 (Some(arg_array), None, false)
//             }
//         } else {
//             (None, Some(1), true)
//         };
//     (res_array, res_scalar, is_res_scalar)
// }

mod tests {
    use  super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::array::{GenericStringArray, StringViewArray};
    use datafusion_expr::ScalarFunctionArgs;
    #[test]
    fn test_regex_instr() {
        test_case_sensitive_regexp_count_scalar();
    }

    fn test_case_sensitive_regexp_count_scalar() {
        let values = ["", "aabca", "abcabc", "abcAbcab", "abcabcabc"];
        let regex = "abc";
        let expected: Vec<i64> = vec![0, 1, 1, 1, 1];

        values.iter().enumerate().for_each(|(pos, &v)| {
            // utf8
            let v_sv = ScalarValue::Utf8(Some(v.to_string()));
            let regex_sv = ScalarValue::Utf8(Some(regex.to_string()));
            let expected = expected.get(pos).cloned();
            let re = RegexpInstrFunc::new().invoke_with_args(ScalarFunctionArgs {
                args: vec![ColumnarValue::Scalar(v_sv), ColumnarValue::Scalar(regex_sv)],
                number_rows: 2,
                return_type: &Int64,
            });
            match re {
                Ok(ColumnarValue::Scalar(ScalarValue::Int64(v))) => {
                    assert_eq!(v, expected, "regexp_instr scalar test failed");
                }
                _ => panic!("Unexpected result"),
            }

            // largeutf8
            let v_sv = ScalarValue::LargeUtf8(Some(v.to_string()));
            let regex_sv = ScalarValue::LargeUtf8(Some(regex.to_string()));
            let re = RegexpInstrFunc::new().invoke_with_args(ScalarFunctionArgs {
                args: vec![ColumnarValue::Scalar(v_sv), ColumnarValue::Scalar(regex_sv)],
                number_rows: 2,
                return_type: &Int64,
            });
            match re {
                Ok(ColumnarValue::Scalar(ScalarValue::Int64(v))) => {
                    assert_eq!(v, expected, "regexp_instr scalar test failed");
                }
                _ => panic!("Unexpected result"),
            }

            // utf8view
            let v_sv = ScalarValue::Utf8View(Some(v.to_string()));
            let regex_sv = ScalarValue::Utf8View(Some(regex.to_string()));
            let re = RegexpInstrFunc::new().invoke_with_args(ScalarFunctionArgs {
                args: vec![ColumnarValue::Scalar(v_sv), ColumnarValue::Scalar(regex_sv)],
                number_rows: 2,
                return_type: &Int64,
            });
            match re {
                Ok(ColumnarValue::Scalar(ScalarValue::Int64(v))) => {
                    assert_eq!(v, expected, "regexp_instr scalar test failed");
                }
                _ => panic!("Unexpected result"),
            }
        });
    }
}