import ast
import time
import logging
import multiprocessing
from termcolor import cprint
from auxiliary_functions import Auxiliary
from input_visitor import InputFunctionVisitor


def run_test(q, r, model_params, num, num_of_processes):
    auxiliary = Auxiliary()
    c = 1
    binary_correctness = []
    params_raised_error = {}
    correct_random_inputs = []
    allowed_time = r.get("allowed_time")
    params_not_raised_error = {}
    model_name = model_params["model"]
    model = auxiliary.rename_model(model_name)
    number_of_parameters = r.get("number_of_parameters")
    allowed_mem_gb = r.get("allowed_memory")
    with open(f"Q{q}/tests.py.template", "r") as f:
        pre_existing_tests_content = f.read()
    num_of_pre_existing_tests = pre_existing_tests_content.count("def")

    while c <= number_of_parameters:
        cprint(
            f"\n{5*'>'} Question {q}, round {num}, instance no. {c}\n",
            "blue",
            attrs=["bold"],
        )
        generated_answer = ""
        correct_random_inputs.clear()
        info_dict = auxiliary.read_file(q, model, c, "info", num, "json")
        function_name = info_dict.get("function_name")

        params = (
            info_dict.get("params")
            or info_dict.get("parameters")
            or info_dict.get("param_set")
            or info_dict.get("param")
            or info_dict.get("args")
        )

        if function_name is None or params is None:
            raise KeyError(
                f"Missing keys in info_dict. Available keys: {list(info_dict.keys())}"
            )






        auxiliary.parameterize_model_answer(q, c, params, model, num)
        generated_response = auxiliary.read_file(q, model, c, "response", num, "txt")
        extract_result, generated_answer = auxiliary.extract_code(generated_response, function_name)

        if extract_result == "No function":
            params_raised_error[c] = (
                params,
                False,
                False,
                "No Python function was generated.",
            )
            binary_correctness.append(0)
            c += 1
            cprint(
                (f"FAIL: No Python function was generated.\n"),
                "red",
                "on_black",
                attrs=["bold"],
            )
            continue

        elif extract_result == "Exception":
            params_raised_error[c] = (
                params,
                False,
                False,
                generated_answer,
            )
            binary_correctness.append(0)
            c += 1
            cprint(
                (f"FAIL: {generated_answer}.\n"),
                "red",
                "on_black",
                attrs=["bold"],
            )
            continue

        all_f_names = auxiliary.create_answer_file(q, c, generated_answer, model, num)
        error_checking_result = auxiliary.checker(q, c, model, num)
        if error_checking_result:
            params_raised_error[c] = (
                params,
                False,
                False,
                f"Linting error: {error_checking_result}",
            )
            binary_correctness.append(0)
            c += 1
            cprint(
                (f"FAIL: {error_checking_result}.\n"),
                "red",
                "on_black",
                attrs=["bold"],
            )
            continue

        f_name = ""
        function_names_equality = False
        if len(all_f_names) == 1:
            if all_f_names[0] == function_name:
                function_names_equality = True
                f_name = all_f_names[0]
        else:
            for name in all_f_names:
                if name == function_name:
                    function_names_equality = True
                    f_name = name
                    break

        if not auxiliary.is_function(generated_answer):
            params_raised_error[c] = (
                params,
                False,
                False,
                "No Python function was generated.",
            )
            binary_correctness.append(0)
            c += 1
            cprint(
                ("FAIL: No Python function was generated.\n"),
                "red",
                "on_black",
                attrs=["bold"],
            )
            continue
        
        tree = ast.parse(generated_answer)
        visitor = InputFunctionVisitor()
        visitor.visit(tree)
        if visitor.found_input:
            params_raised_error[c] = (
                params,
                False,
                False,
                "The generated response prompts the user for input.",
            )
            binary_correctness.append(0)
            c += 1
            cprint(
                ("FAIL: The generated response prompts the user for input.\n"),
                "red",
                "on_black",
                attrs=["bold"],
            )
            continue

        if not function_names_equality:
            params_raised_error[c] = (
                params,
                False,
                False,
                "Wrong function name.",
            )
            binary_correctness.append(0)
            c += 1
            cprint(
                ("FAIL: Wrong function name.\n"),
                "red",
                "on_black",
                attrs=["bold"],
            )
            continue
        
        try:
            comparison_result = auxiliary.compare_param_nums(q, c, function_name, f_name, model, params, num)
        except Exception as e:
            params_raised_error[c] = (
                params,
                False,
                False,
                str(e),
            )
            binary_correctness.append(0)
            c += 1
            cprint(
                (f"{e}\n"),
                "red",
                "on_black",
                attrs=["bold"],
            )
            continue

        if not comparison_result:
            params_raised_error[c] = (
                params,
                False,
                False,
                "Wrong number of parameters in the function signature.",
            )
            binary_correctness.append(0)
            c += 1
            cprint(
                ("FAIL: Wrong number of parameters in the function signature.\n"),
                "red",
                "on_black",
                attrs=["bold"],
            )
            continue

        auxiliary.parameterize_tests(q, c, f_name, params, model, num)
        auxiliary.create_pytest_ini_file(q, c, model, num, allowed_time, num_of_processes)
        auxiliary.create_conftest_file(q, c, model, num)
        pytest_params = [
            "-v",
            f"Q{q}/{model}_results_{num}/Folder_{c}/tests{q}_{c}_{num}.py",
            "-c",
            f"Q{q}/{model}_results_{num}/Folder_{c}/Q{q}_{c}.ini",
        ]

        pytest_result_queue = multiprocessing.Queue()
        monitor_result_queue = multiprocessing.SimpleQueue()
        pytest_start_time = time.time()
        pytest_process = multiprocessing.Process(target=auxiliary.run_pytest, args=(pytest_params, pytest_result_queue))
        pytest_process.start()
        pytest_process_id = pytest_process.pid

        monitor_pytest = multiprocessing.Process(
            target=auxiliary.monitor_time_space,
            args=(pytest_process_id, allowed_mem_gb, pytest_start_time, allowed_time * num_of_pre_existing_tests, monitor_result_queue),
            daemon=True
            )
        monitor_pytest.start()
        monitor_pytest.join()
        monitor_pytest_result = monitor_result_queue.get()

        if monitor_pytest_result != "done":
            try:
                auxiliary.force_kill_process(pytest_process)
            except Exception as e:
                logging.error(f"Error checking process status: {e}")
                    
            params_raised_error[c] = (
                    params,
                    True,
                    False,
                    "Resource exhaustion",
                )
            binary_correctness.append(0)
            c += 1
            cprint(
                ("FAIL: Resource exhaustion during pre-existing tests.\n"),
                "red",
                "on_black",
                attrs=["bold"],
            )
            continue   
        else:
            pytest_process.join(timeout=1)
            pytest_results, exception_result = None, None
            if not pytest_result_queue.empty():
                (pytest_results, exception_result) = pytest_result_queue.get()
            if exception_result:
                params_raised_error[c] = (
                    params,
                    False,
                    False,
                    exception_result,
                )
                binary_correctness.append(0)
                c += 1
                cprint(
                    (f"FAIL: {exception_result}\n"),
                    "red",
                    "on_black",
                    attrs=["bold"],
                )
                continue
            else:
                auxiliary.write_test_cases_json_report(q, model, c, params, num)
                if pytest_results != 0:
                    params_raised_error[c] = (
                        params,
                        True,
                        False,
                        "Check the pre-existing tests report.",
                    )
                    binary_correctness.append(0)
                    c += 1
                    cprint(
                        (f"FAILED during pre-existing tests\n"),
                        "red",
                        "on_black",
                        attrs=["bold"],
                    )
                    continue

        code_is_correct = True
        failure_reason = ""
        if r.get("fuzzy_testing").lower() == "true":
            cprint(
                ("\nStart fuzzy testing...\n"),
                "light_blue",
                "on_black",
                attrs=["bold"],
            )
            auxiliary.setup_logging(f"Q{q}/{model}_results_{num}/Folder_{c}/process_termination.log")
            number_of_random_inputs = r.get("number_of_fuzzy_inputs")
            generated_answer_result, model_answer_result = "", ""
            counter_examples = {}
            for i in range(number_of_random_inputs):
                cprint(
                    f"\nRandom input no. {i + 1}",
                    "white",
                    "on_black",
                    attrs=["bold"],
                )
                r_i = auxiliary.get_random_input(q, params, i)
                random_input = None
                if q == 34:
                    random_input = r_i
                elif isinstance(r_i, tuple) and q not in [48, 54, 59, 60]:
                    random_input = [*r_i]
                else:
                    random_input = [r_i]

                result_queue = multiprocessing.Queue()
                simple_queue = multiprocessing.SimpleQueue()
                start_time = time.time()
                proc = multiprocessing.Process(
                    target=auxiliary.get_generated_answer_result,
                    args=(
                        q,
                        c,
                        f_name,
                        model,
                        random_input,
                        num,
                        result_queue,
                    ),
                )
                proc.start()
                pid = proc.pid

                monitor = multiprocessing.Process(
                    target=auxiliary.monitor_time_space,
                    args=(pid, allowed_mem_gb, start_time, allowed_time, simple_queue),
                    daemon=True
                )
                monitor.start()
                monitor.join()
                monitor_process_result = simple_queue.get()
                    
                equality_of_answers = False

                if monitor_process_result != "done":
                    try:
                        auxiliary.force_kill_process(proc)
                    except Exception as e:
                        logging.error(f"Error checking process status: {e}")
                    
                    code_is_correct = False
                    failure_reason = "Resource exhaustion"
                    cprint(
                        ("FAIL: Resource exhaustion.\n"),
                        "red",
                        "on_black",
                        attrs=["bold"],
                    )
                    counter_examples = auxiliary.add_counter_example(q, random_input, None, None)
                    break
                else:
                    proc.join()
                    if not result_queue.empty():
                        generated_answer_result = result_queue.get()

                if isinstance(generated_answer_result, str) and generated_answer_result.startswith("exception"):
                    code_is_correct = False
                    failure_reason = f"LLM's code raised {generated_answer_result}"
                    cprint(
                        (f"FAIL: {generated_answer_result}\n"),
                        "red",
                        "on_black",
                        attrs=["bold"],
                    )
                    counter_examples = auxiliary.add_counter_example(q, random_input, None, None)
                    break

                equality_of_answers = False
                model_answer_result = auxiliary.get_model_answer_result(
                    q, c, function_name, model, random_input, num
                )

                if type(model_answer_result) == list:
                    if q == 54 or q == 59 or q == 60:
                        equality_of_answers = (
                        generated_answer_result == model_answer_result
                    )
                    elif q == 48:
                        equality_of_answers = (
                        generated_answer_result.upper() == model_answer_result
                    )
                    elif q == 57:
                        equality_of_answers = auxiliary.if_two_lists_of_matrices_are_equal(
                        generated_answer_result, model_answer_result
                    )
                    elif q == 55 or q == 56:
                        equality_of_answers = (
                        auxiliary.if_two_lists_are_equal_order_irrelevant(
                            generated_answer_result, model_answer_result
                        )
                    )
                    else:
                        equality_of_answers = auxiliary.if_two_lists_are_equal(
                        generated_answer_result, model_answer_result
                    )
                else:
                    equality_of_answers = (
                    generated_answer_result == model_answer_result
                )

                if not equality_of_answers:
                    code_is_correct = False
                    failure_reason = "Check fuzzy test report."
                    cprint(
                        ("Failed during fuzzy test.\n"),
                        "red",
                        "on_black",
                        attrs=["bold"],
                    )
                    counter_examples = auxiliary.add_counter_example(q, random_input, model_answer_result, generated_answer_result)
                    break
                else:
                    code_is_correct = True
                    correct_random_inputs.append(random_input)
                    cprint("Pass.", "green", "on_black", attrs=["bold"])
                    
            auxiliary.extra_testing_report(
                q,
                c,
                model,
                num,
                counter_examples,
                correct_random_inputs,
            )
        
        if code_is_correct:
            auxiliary.create_visual_tree(q, c, model, num, True)
            auxiliary.create_string_tree(q, c, model, num, True)
            params_not_raised_error[c] = params
            binary_correctness.append(1)
        else:
            params_raised_error[c] = (
                        params,
                        False,
                        True,
                        failure_reason,
                    )
            binary_correctness.append(0)
        
        c += 1

    if params_raised_error:
        auxiliary.wrong_params_failure_reasons_writer(q, model, params_raised_error, num)

    auxiliary.param_report_file_writer(
        q,
        params_raised_error,
        params_not_raised_error,
        model,
        model_name,
        num,
    )

    return binary_correctness

