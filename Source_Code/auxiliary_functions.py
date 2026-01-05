import os
import re
import ast
import sys
import time
import json
import psutil
import pytest
import signal
import logging
import platform
import numpy as np
import importlib.util
from io import StringIO
from pathlib import Path
from pylint.lint import Run
from termcolor import cprint
from inspect import signature
from test_plugin import TestResults
from graphviz import Source, Digraph
from anytree import Node, RenderTree
from pylint.reporters import JSONReporter



class Auxiliary():
    def __init__(self):
        pass


    def add_counter_example(self, q, random_input, model_answer_result, generated_answer_result):
        counter_examples = {}
        if q == 41:
            counter_examples[
                tuple(
                    tuple(
                        (random_input[0])
                        + ["1000000000000000000001"]
                        + (random_input[1])
                    )
                )
            ] = (model_answer_result, generated_answer_result)

        elif q in [
            22,
            23,
            24,
            28,
            30,
            32,
            33,
            35,
            36,
            37,
            39,
            40,
            45,
            47,
            49,
            52,
            53,
            56,
        ]:
            counter_examples[tuple(random_input)] = (
                model_answer_result,
                generated_answer_result,
            )

        elif q == 34:
            c_exa = []
            for x in random_input:
                c_exa.append(frozenset(x))
            counter_examples[tuple(c_exa)] = (
                model_answer_result,
                generated_answer_result,
            )

        elif q == 57 or q == 58:
            counter_examples[str(*random_input)] = (
                model_answer_result,
                generated_answer_result,
            )

        else:
            counter_examples[tuple(*random_input)] = (
                model_answer_result,
                generated_answer_result,
            )

        return counter_examples


    def prepare_data(self, data, m=100, r=5):
        list_of_hundreds = []
        while data:
            list_of_hundreds.append(data[:m])
            data = data[m:]
        final_list = []
        for i in range(m):
            list_of_fives = []
            for j in range(r):
                list_of_fives.append(list_of_hundreds[j][i])
            final_list.append(list_of_fives)
        
        return final_list


    
    def calculate_accuracy(self, data, R=None, M=None, *args):
        s = 0
        for lst in data:
            s += sum(lst)

        R = len(data[0]) if data and data[0] else 0
        M = len(data)

        return (s / (R * M)) if (R * M) != 0 else 0.0



    def calculate_correctness_potential(self, data, M=None, *args):
        """
        Correctness Potential (CPS):
        fraction of parameter settings for which
        at least one round succeeds
        """
        if not data:
            return 0.0

        s = 0
        for lst in data:
            if sum(lst) >= 1:
                s += 1

        M = len(data)
        return s / M

    

    def calculate_consistent_correctness(self, data, R=None, M=None, *args):
        """
        Consistent Correctness (CCS):
        fraction of parameter settings for which
        all rounds succeed
        """
        if not data or not data[0]:
            return 0.0

        s = 0
        R = len(data[0])

        for lst in data:
            if sum(lst) == R:
                s += 1

        M = len(data)
        return s / M





    def checker(self, q_no, c, api_name, no):
        reporter = StringIO()
        Run(
            [f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/generated_answer.py"],
            reporter=JSONReporter(reporter),
            exit=False,
        )
        results = json.loads(reporter.getvalue())
        for i in results:
            if i.get("message-id").startswith("E"):
                return i.get("message")
        return None


    def compare_param_nums(self, q_no, c, fun_name, f_name, api_name, params, no):
        solution_module = self.import_preparation(
            "solution", f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/solution.py"
        )
        solution_module_function = getattr(solution_module, fun_name)

        generated_answer_module = self.import_preparation(
            "generated_answer",
            f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/generated_answer.py",
        )
        generated_answer_module_function = getattr(generated_answer_module, f_name)
        solution_sig = signature(solution_module_function)
        generated_answer_sig = signature(generated_answer_module_function)

        if q_no == 34:
            if (
                len(generated_answer_sig.parameters) == int(params[0])
                or len(generated_answer_sig.parameters) == 1
            ):
                return True
            else:
                return False

        return len(solution_sig.parameters) == len(generated_answer_sig.parameters)


    def create_answer_file(self, q_no, c, answer, api_name, no):
        with open(
            f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/generated_answer.py", "w"
        ) as f:
            f.write(answer)
            function_names = self.get_all_function_names(answer)
            return function_names


    def create_conftest_file(self, q_no, c, api_name, no):
        with open(f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/conftest.py", "w") as f:
            content = (
                "import platform\n\n\ndef "
                f'pytest_html_report_title(report):\n    report.title = "Question {q_no} with parameters set no. '
                f'{c}"\n\n\ndef pytest_configure(config):\n    config._metadata = '
                '{\n        "Benchmarking framework": "Turbulence",\n        "Operating system": platform.platform(),'
                '\n        "Python version": platform.python_version()\n    }\n\n\n'
                "def pytest_html_results_table_header(cells):\n    cells.pop()\n\n\n"
                "def pytest_html_results_table_row(cells):\n    cells.pop()"
            )
            f.write(content)


    def create_init_py_file(self, q_no, c, api_name, no):
        with open(f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/__init__.py", "w") as _:
            pass


    def create_model_folder(self, path, api_name, no):
        full_path = path + f"/{api_name}_results_{no}"
        Path(full_path).mkdir(parents=True, exist_ok=True)


    def create_result_folder(self, path, c, api_name, no):
        full_path = path + f"/{api_name}_results_{no}/Folder_{c}"
        Path(full_path).mkdir(parents=True, exist_ok=True)


    def create_pytest_ini_file(self, q_no, c, api_name, no, timeout, num_of_processes):
        with open(f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/Q{q_no}_{c}.ini", "w") as f:
            content = f"[pytest]\naddopts = -p no:warnings -n {num_of_processes} --maxfail=1 --timeout={timeout} --report-log=Q{q_no}/{api_name}_results_{no}/Folder_{c}/test_report.json"
            f.write(content)


    def create_string_tree(self, q_no, c, api_name, no, inside_folder):
        with open(
            f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/generated_answer.py", "r"
        ) as f:
            code = f.read()
        tree = ast.parse(code)
        root = Node("Root")

        def create_tree(node, parent):
            ast_node = Node(str(node.__class__.__name__), parent=parent)
            for child in ast.iter_child_nodes(node):
                create_tree(child, ast_node)

        create_tree(tree, root)

        if inside_folder:
            location = f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/text_tree.txt"
        else:
            location = f"Q{q_no}/{api_name}_results_{no}/common_visual_tree.txt"

        with open(location, "w") as f:
            for x, _, node in RenderTree(root):
                f.write("%s%s\n" % (x, node.name))


    def create_visual_tree(self, q_no, c, api_name, no, inside_folder):
        with open(
            f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/generated_answer.py", "r"
        ) as f:
            code = f.read()
        tree = ast.parse(code)

        def create_graph(node, graph):
            node_name = str(node.__class__.__name__)

            # colours available at https://graphviz.org/doc/info/colors.html
            graph.node(
                str(node),
                label=node_name,
                shape="box",
                style="filled, rounded",
                fillcolor="aquamarine3",
            )
            for child in ast.iter_child_nodes(node):
                child_name = str(child.__class__.__name__)
                graph.node(str(child), label=child_name)
                graph.edge(str(node), str(child))
                create_graph(child, graph)

        graph = Digraph()
        create_graph(tree, graph)
        src = Source(graph.source)
        if inside_folder:
            location = f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/visual_tree"
        else:
            location = f"Q{q_no}/{api_name}_results_{no}/common_visual_tree"
        src.render(
            location,
            format="svg",
            view=False,
        )


    def extract_code(self, txt_string, function_name):
        if txt_string.count("```") < 2:
            import_lines = []
            other_lines = []
            txt_string = txt_string.encode().decode('unicode_escape')
            txt_string = txt_string.replace("python", "").replace("Python", "")
            # txt_string = txt_string.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '    ').replace("\\'",
            #                                                                                                 "'").replace(
            #     "``", "").replace("```", "").replace("python", "").strip("\n")
            if not txt_string:
                return "No function", ""

            lines = txt_string.splitlines()

            for i, line in enumerate(lines):
                if f"def {function_name}" in line:
                    lines[i] = line[line.find("def"):]
                    break

            lines_of_code = [line for line in lines if
                            (line.startswith(
                                ' ') and ("\\'\\'\\'" not in line)) or 'import' in line or 'from' in line or 'def' in line]

            for line in lines_of_code:
                stripped_line = line.strip()
                if stripped_line.startswith('import ') or stripped_line.startswith('from '):
                    import_lines.append(line)
                else:
                    if f"def {function_name}" in stripped_line:
                        def_idx = line.find("def")
                        other_lines.append(line[def_idx:])
                    else:
                        other_lines.append(line)

            desired_idx = -1
            for idx in range(len(other_lines)):
                if other_lines[idx].startswith("def"):
                    desired_idx = idx
                if idx == len(other_lines) - 1:
                    if other_lines[idx].count('"') == 1:
                        other_lines[idx] = other_lines[idx].replace('"', '')
                    if other_lines[idx].count('`') <= 3:
                        other_lines[idx] = other_lines[idx].replace('`', '')
                    if other_lines[idx].count("'") == 1:
                        other_lines[idx] = other_lines[idx].replace("'", "")

            if desired_idx == -1:
                return "No function", ""
            else:
                other_lines = other_lines[desired_idx:]
            code_without_imports = '\n'.join(other_lines)

            try:
                tree = ast.parse(code_without_imports)
                func_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
                if not func_defs:
                    return "No function", ""
                functions_code = ''
                for func in func_defs:
                    functions_code += ast.unparse(func) + '\n\n'
                output_code = '\n'.join(import_lines) + '\n\n' + functions_code.strip()
                return "Function", output_code.strip()
            except Exception as e:
                return "Exception", str(e)
        else:
            indices = [match.start() for match in re.finditer(r'```', txt_string)]
            for k in range(len(indices)):
                for j in range(k + 1, len(indices)):
                    first_idx = indices[k]
                    second_idx = indices[j]
                    sliced_txt = txt_string[first_idx + 3: second_idx]
                    sliced_txt = sliced_txt.encode().decode('unicode_escape')
                    generated_code = sliced_txt.replace("python", "").replace("Python", "")
                    # generated_code = sliced_txt.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'").replace('\\t',
                    #                                                                                                 '    ').replace(
                    #     "python", "").replace("``", "").replace("```", "").strip("\n")
                    if not generated_code:
                        break
                    else:
                        lines = generated_code.splitlines()
                        for i, line in enumerate(lines):
                            if line.lstrip().startswith('def'):
                                lines[i] = line.lstrip()
                                break

                        import_lines = []
                        other_lines = []

                        for line_idx in range(len(lines)):
                            if lines[line_idx].startswith("`") or lines[line_idx].strip().startswith("`"):
                                backtick_idx = lines[line_idx].find("`")
                                lines[line_idx] = lines[line_idx][backtick_idx + 1:]
                            if lines[line_idx].startswith("``") or lines[line_idx].strip().startswith("``"):
                                backtick_idx = lines[line_idx].find("``")
                                lines[line_idx] = lines[line_idx][backtick_idx + 2:]
                            if lines[line_idx].startswith("```") or lines[line_idx].strip().startswith("```"):
                                backtick_idx = lines[line_idx].find("```")
                                lines[line_idx] = lines[line_idx][backtick_idx + 3:]

                        for line in lines:
                            stripped_line = line.strip()
                            if stripped_line.startswith('import ') or stripped_line.startswith('from '):
                                import_lines.append(line)
                            else:
                                other_lines.append(line)

                        for idx in range(len(other_lines)):
                            if idx == len(other_lines) - 1:
                                if other_lines[idx].count('"') == 1:
                                    other_lines[idx] = other_lines[idx].replace('"', '')
                                if other_lines[idx].count('`') <= 3:
                                    other_lines[idx] = other_lines[idx].replace('`', '')
                                if other_lines[idx].count("'") == 1:
                                    other_lines[idx] = other_lines[idx].replace("'", "")

                        code_without_imports = '\n'.join(other_lines)

                        if "def" not in code_without_imports:
                            break

                        try:
                            tree = ast.parse(code_without_imports)
                            func_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
                            if not func_defs:
                                break

                            functions_code = ''
                            for func in func_defs:
                                functions_code += ast.unparse(func) + '\n\n'
                            output_code = '\n'.join(import_lines) + '\n\n' + functions_code.strip()
                            return "Function", output_code.strip()
                        except Exception as e:
                            return "Exception", str(e)
            return "No function", ""


    def extract_questions_params_inputs(self, questions_params_inputs):
        q_requirements = {}
        for k, v in questions_params_inputs.items():
            if "," in k:
                comma_split = [i.strip() for i in k.split(",")]
                for x in comma_split:
                    if "-" in x:
                        dash_index = x.find("-")
                        start = int(x[:dash_index])
                        stop = int(x[dash_index + 1 :])
                        for i in range(start, stop + 1):
                            q_requirements[i] = v
                    else:
                        q_requirements[int(x)] = v
            elif "-" in k:
                dash_index = k.find("-")
                start = int(k[:dash_index])
                stop = int(k[dash_index + 1 :])
                for i in range(start, stop + 1):
                    q_requirements[i] = v
            else:
                q_requirements[int(k)] = v

        return dict(sorted(q_requirements.items()))


    def extract_result_folder_numbers(self, result_folder):
        nums = []
        if "," in result_folder:
            comma_split = [i.strip() for i in result_folder.split(",")]
            for x in comma_split:
                if "-" in x:
                    dash_index = x.find("-")
                    start = int(x[:dash_index])
                    stop = int(x[dash_index + 1 :])
                    for i in range(start, stop + 1):
                        nums.append(int(i))
                else:
                    nums.append(int(x))
        elif "-" in result_folder:
            dash_index = result_folder.find("-")
            start = int(result_folder[:dash_index])
            stop = int(result_folder[dash_index + 1 :])
            for i in range(start, stop + 1):
                nums.append(int(i))
        else:
            nums.append(int(result_folder.strip()))

        return sorted(nums)


    def extra_testing_report(
        self,
        q_no,
        c,
        api_name,
        no,
        counter_example,
        correct_random_inputs,
    ):
        with open(
            f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/fuzzy_test_report.txt", "w"
        ) as f:
            f.write(
                f"Out of {len(correct_random_inputs) + len(counter_example)} random test input(s), the code returned by LLM passed {len(correct_random_inputs)} times and failed {len(counter_example)} times.\n\n"
            )
            c = 1
            if counter_example:
                f.write("The counter examples were as follows:\n\n")
                for example, answers in counter_example.items():
                    if q_no == 57:
                        f.write(f"({c}) With the following counter example:\n{example}\n\n")
                        
                    elif q_no == 41:
                        f.write(f"({c}) With the following counter examples:\n\n")
                        seperator_idx = example.index("1000000000000000000001")
                        f.write(f"Input 1: {list(example[:seperator_idx])}\n")
                        f.write(f"Input 2: {list(example[seperator_idx + 1:])}\n\n")
                    elif q_no == 57 or q_no == 58:
                        f.write(f"({c}) With the following counter example:\n{example}\n\n")
                    else:
                        f.write(f"({c}) With the following counter example:\n{list(example)}\n\n")
                    
                    if answers[0] is not None:
                        f.write("\nThe correct answer was:\n\n")
                        if q_no == 57:
                            f.write(f"{answers[0]}\n\nBut the LLM code returned:\n{answers[1]}\n\n")
                        else:
                            f.write(f"{answers[0]}\n\nBut the LLM code returned:\n{answers[1]}\n\n")
                           
                    f.write("=" * 150)
                    f.write("\n\n")
                    c += 1

            if correct_random_inputs:
                if q_no == 41:
                    inp = "inputs"

                else:
                    correct_random_inputs = self.flatten_list(correct_random_inputs)
                    inp = "input"

                f.write(
                    f"The test {inp} with which the LLM code passed were as follows:\n\n"
                )
                for random_input in correct_random_inputs:
                    f.write(f"({c}) Test {inp}:\n\n")
                    if q_no == 41:
                        f.write(f"Input 1: {random_input[0]}\n")
                        f.write(f"Input 2: {random_input[1]}\n\n")
                    elif q_no == 57:
                        f.write(f"{random_input}\n\n")
                    else:
                        f.write(f"{random_input}\n\n")
                    f.write("=" * 150)
                    f.write("\n\n")
                    c += 1


    def flatten_list(self, given_list):
        return [element for sublist in given_list for element in sublist]


    def get_all_function_names(self, text):
        matches = re.findall(r"def\s+(\w+)\(", text)
        return matches


    def get_generated_answer_result(self, q_no, c, fun_name, api_name, input, no, result_queue):  
        try:
            module = self.import_preparation(
            "generated_answer",
            f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/generated_answer.py",
        )
            generated_solution = getattr(module, fun_name)
            result = generated_solution(*input)
            
        except Exception as e:
            print(f"Exception in the LLM's generated code: {e}")
            result_queue.put(f'exception:{e}')

        else:
            result_queue.put(result)
    

    def get_model_answer_result(self, q_no, c, fun_name, api_name, input, no):
        module = self.import_preparation(
            "solution", f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/solution.py"
        )
        model_solution = getattr(module, fun_name)
        result = model_solution(*input)
        return result


    def get_params(self, q_no, seed):
        module = self.import_preparation("genparams", f"Q{q_no}/genparams.py")
        return module.gen_params(q_no, seed)


    def get_random_input(self, q_no, l, i):
        module = self.import_preparation(
            "gen_function_params", f"Q{q_no}/gen_function_params.py"
        )
        return module.input_generator(l, i)


    def get_rank(self, num):
        if 1 <= num <= 5:
            return 'A'
        elif 6 <= num <= 10:
            return 'B'
        elif 11 <= num <= 20:
            return 'C'
        elif 21 <= num <= 30:
            return 'D'
        elif 31 <= num <= 40:
            return 'E'
        else:
            return 'F'
    

    def if_two_lists_are_equal(self, l1, l2):
        if len(l1) != len(l2):
            return False
        for i in range(len(l1)):
            if l1[i] != l2[i]:
                return False
        return True


    def if_two_lists_are_equal_order_irrelevant(self, l1, l2):
        if len(l1) != len(l2):
            return False
        else:
            for i in l1:
                if l1.count(i) != l2.count(i):
                    return False
        return True


    def if_two_lists_of_matrices_are_equal(self, l1, original_l2):
        l2 = [np.array(m) for m in original_l2]
        if len(l1) != len(original_l2):
            return False
        for m1 in l1:
            c = len(l2)
            pointer = 0
            while l2 and c > 0:
                m2 = l2[pointer]
                if np.array_equal(m1, m2):
                    del l2[pointer]
                else:
                    pointer += 1
                c -= 1

        return False if l2 else True


    def equality_of_lists_of_lists_of_lists(self, l1, l2):
        i = len(l1) * len(l2)
        c = 0
        while l1 and c <= i:
            j = 0
            while l2 and j < len(l2):
                x, y = l1[0], l2[j]
                if self.if_two_lists_are_equal_order_irrelevant(x, y):
                    l1.remove(x)
                    l2.remove(y)
                else:
                    j += 1
            c += 1
        if l1 or l2:
            return False
        else:
            return True
        
    
    def extract_info_from_pytest(self, data):
        results = {}
        for line in data.strip().split('\n'):
            record = json.loads(line)
            if "nodeid" in record:
                x = record["nodeid"]
                if '::' in x:
                    idx = x.find('::')
                    x = x[idx + 2:]
                    outcome = record["outcome"]
                    duration = record["duration"]
                    if x not in results:
                        if outcome == 'failed':
                            if type(record["longrepr"]) != dict:
                                results[x] = (record["outcome"], record["longrepr"], round(duration, 5))
                            else:
                                results[x] = (record["outcome"], record["longrepr"]["reprcrash"]["message"], round(duration, 5))
                        else:
                            results[x] = (record["outcome"], '', round(duration, 5))
                    else:
                        if results[x][0] != outcome:
                            if outcome == 'failed':
                                if type(record["longrepr"]) != dict:
                                    results.update({x: (record["outcome"], record["longrepr"], round(duration, 5))})
                                else:
                                    results.update({x: (record["outcome"], record["longrepr"]["reprcrash"]["message"], round(duration, 5))})
        return results


    def find_file_name(self, directory, part_of_name):
        files_in_directory = os.listdir(directory)
        desired_files = [file for file in files_in_directory if file.endswith(part_of_name)]

        return desired_files[0]
    

    def force_kill_process(self, proc):
        if proc.is_alive():
            cprint("Attempting to terminate the process...", "yellow")
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                cprint("Process did not terminate. Killing process...", "yellow")
                proc.kill()
                proc.join(timeout=5)
                if proc.is_alive():
                    cprint("Warning: Process could not be killed. Proceeding anyway.", "yellow")
                    self.log_unkillable_process(proc.pid)


    def import_preparation(self, name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


    def instantiate(self, template, args):
        for i in range(len(args)):
            template = template.replace(f"${i}", str(args[i]))

        return template


    def is_function(self, answer):
        try:
            tree = ast.parse(answer)
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return True
        return False
    

    def kill_proc_tree(self, pid, including_parent, timeout=0.5):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                child.kill()
            
            start_time = time.time()
            while children:
                _, alive = psutil.wait_procs(children, timeout=0.1)
                if not alive or (time.time() - start_time) > timeout:
                    break
                children = alive

            for p in alive:
                p.kill()
            
            if including_parent:
                parent.kill()
                try:
                    parent.wait(timeout=timeout)
                except psutil.TimeoutExpired:
                    parent.kill()
        except psutil.NoSuchProcess:
            pass


    def log_unkillable_process(self, proc_pid):
        try:
            parent = psutil.Process(proc_pid)
            cmdline = parent.cmdline()
            status = parent.status()
            create_time = parent.create_time()
            memory_info = parent.memory_info()
            
            logging.error(f"Could not kill process {proc_pid}.")
            logging.error(f"Command line: {' '.join(cmdline)}")
            logging.error(f"Status: {status}")
            logging.error(f"Create time: {time.ctime(create_time)}")
            logging.error(f"Memory info: RSS={memory_info.rss}, VMS={memory_info.vms}")
            
            children = parent.children(recursive=True)
            for child in children:
                child_cmdline = ' '.join(child.cmdline())
                child_status = child.status()
                logging.error(f"Child Process PID: {child.pid}, Command line: {child_cmdline}, Status: {child_status}")
        except Exception as e:
            logging.error(f"Failed to retrieve process details for PID {proc_pid}: {e}")

    
    def monitor_time_space(self, pid, allowed_mem_gb, start_time, allowed_time, q):
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            q.put("done")
            return

        while True:
            try:
                if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                    q.put("done")
                    break

                # Check process memory usage in bytes
                memory_usage = proc.memory_info().rss
                # Include memory usage of all child processes
                children = proc.children(recursive=True)
                for child in children:
                    try:
                        memory_usage += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue  # Child process may have terminated
                if memory_usage > allowed_mem_gb * (10 ** 9):
                    self.kill_proc_tree(pid, True, 0.1)
                    q.put("High memory usage.")
                    cprint("High memory usage.\n", "red", "on_black", attrs=["bold"])
                    break

                # Check execution time
                if time.time() - start_time > allowed_time:
                    self.kill_proc_tree(pid, True, 0.1)
                    q.put("High execution runtime.")
                    cprint("High execution runtime.\n", "red", "on_black", attrs=["bold"])
                    break

                time.sleep(0.05)

            except psutil.NoSuchProcess:
                q.put("done")
                break
            except Exception as e:
                q.put(str(e))
                cprint(f"{e}\n", "red", "on_black", attrs=["bold"])
                break


    def parameterize_model_answer(self, q_no, c, params, api_name, no):
        with open(f"Q{q_no}/solution.py.template", "r") as f0, open(
            f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/solution.py", "w"
        ) as f1:
            solution_template = f0.read()
            solution = self.instantiate(solution_template, params)
            f1.write(solution)
            return solution


    def parameterize_question(self, q_no, params):
        with open(f"Q{q_no}/question.txt.template", "r") as f:
            question_template = f.readline().strip("\n")
            function_name = re.search(r"(?<=').*?(?=')", question_template).group(0)
            question = self.instantiate(question_template, params)
            return question, function_name


    def parameterize_tests(self, q_no, c, function_name, params, api_name, no):
        with open(f"Q{q_no}/tests.py.template", "r") as f0, open(
            f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/tests{q_no}_{c}_{no}.py", "w"
        ) as f1:
            test_template = f0.read()
            modified_params = params.copy()
            modified_params.append(c)
            modified_params.append(function_name)
            modified_params.append(api_name)
            modified_params.append(no)
            f1.write(self.instantiate(test_template, modified_params))


    def param_report_file_writer(
        self,
        q_no,
        params_raised_error,
        params_not_raised_error,
        api_name,
        model_name,
        no,
    ):
        main_dict = {"model_name": model_name}
        if not params_raised_error:
            main_dict["wrong"] = 0
        else:
            wrong_dict = {}
            for f_n, v in params_raised_error.items():
                wrong_dict[f"Folder_{f_n}"] = v[0]
            main_dict["wrong"] = wrong_dict

        if not params_not_raised_error:
            main_dict["correct"] = 0
        else:
            correct_dict = {}
            for f_n, p in params_not_raised_error.items():
                correct_dict[f"Folder_{f_n}"] = p
            main_dict["correct"] = correct_dict

        with open(f"Q{q_no}/{api_name}_results_{no}/params_report.json", "w") as f:
            json.dump(main_dict, f, ensure_ascii=False, indent=2)

        content_header = f"""
        <!DOCTYPE html>
        <html>
        <body>\n
        """
        if params_not_raised_error:
            concat = """
            <h1>Parameters with correct generated code</h1>
            <table style="border: 2px solid black; border-collapse:collapse">
                <tr style="background-color:green; color:white">
                    <th width="75" style="border: 1px solid black; text-align: center">Folder Number</th>
                    <th width="150" style="border: 1px solid black; text-align: center">Parameters</th>
                    <th width="350" style="border: 1px solid black; text-align: center">Link of Pre-Existing Tests Reports</th>
                    <th width="350" style="border: 1px solid black; text-align: center">Link of Fuzzy Tests Reports</th>
                </tr>
            """
            for folder, tup in params_not_raised_error.items():
                cells = f"""
                <tr>
                    <td style="border: 1px solid black; text-align: center">
                    <a href="Folder_{folder}" target="_blank">{folder}</a>
                    </td>
                    <td style="border: 1px solid black; text-align: center">
                    {tup[0]}
                    </td>
                    <td style="border: 1px solid black";>
                    <a href="Folder_{folder}/pre_existing_test_case_report.json" target="_blank"> Click to see pre-existing test report. </a>
                    </td>
                    <td style="border: 1px solid black";>
                    <a href="Folder_{folder}/fuzzy_test_report.txt" target="_blank"> Click to see fuzzy test report.</a>
                    </td>
                    </td>
                </tr>\n
                """
                concat += cells
            concat += "</table>"

            with open(f"Q{q_no}/{api_name}_results_{no}/correct_params.html", "w") as f:
                f.write(content_header + concat)

        if params_raised_error:
            concat = f"""
            <h1>Parameters with wrong generated code</h1>
            <table style="border: 2px solid black; border-collapse:collapse">
                <tr style="background-color:FireBrick; color:white">
                    <th width="200" style="border: 1px solid black; text-align: left">Folder Number</th>
                    <th width="200" style="border: 1px solid black; text-align: left">Parameters</th>
                    <th width="400" style="border: 1px solid black; text-align: left">Link of Pre-Existing Tests Reports</th>
                    <th width="420" style="border: 1px solid black; text-align: left">Link of Fuzzy Tests Reports</th>
                    <th width="400" style="border: 1px solid black; text-align: left">Test Failure Reason</th>
                </tr>
            """
            for folder, tup in params_raised_error.items():
                
                test_cases_report = (
                    f'<a href="Folder_{folder}/pre_existing_test_case_report.json" target="_blank"> Click to see pre-existing test report. </a>'
                    if tup[1]
                    else ""
                )
                fuzzy_report = (
                    f'<a href="Folder_{folder}/fuzzy_test_report.txt" target="_blank"> Click to see fuzzy test report.</a>'
                    if tup[2]
                    else ""
                )
                cells = f"""
                <tr>
                    <td style="border: 1px solid black";>
                    <a href="Folder_{folder}" target="_blank">{folder}</a>
                    </td>
                    <td style="border: 1px solid black";>
                    {tup[0]}
                    </td>
                    <td style="border: 1px solid black";>
                    {test_cases_report}
                    </td>
                    <td style="border: 1px solid black";>
                    {fuzzy_report}
                    </td>
                    <td style="border: 1px solid black";>
                    {tup[-1]}
                    </td>
                </tr>\n
                """
                concat += cells

            concat += "</table>"

            with open(f"Q{q_no}/{api_name}_results_{no}/wrong_params.html", "w") as f:
                f.write(content_header + concat)


    def read_file(self, q_no, api_name, c, file_name, no, extention):
        with open(
            f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/{file_name}.{extention}", "r"
        ) as f:
            if extention == "json":
                return json.load(f)
            return f.read()


    def rename_model(self, api_name: str):
        if api_name == "claude-3-5-sonnet-20240620":
            return "claude_3_5_sonnet"
        elif api_name == "claude-3-5-haiku-20241022":
            return "claude_3_5_haiku"
        elif api_name == "codegemma-7b-it":
            return "codegemma"
        elif api_name == "codestral-latest":
            return "codestral"
        elif api_name == "command-r-plus-08-2024":
            return "command_r_plus"
        elif api_name == "databricks/dbrx-instruct":
            return "dbrx"
        elif api_name == "deepseek-coder":
            return "deepseek-coder"
        elif api_name == "cognitivecomputations/dolphin-mixtral-8x22b":
            return "dolphin2_9_2"
        elif api_name == "gemini-1.5-pro":
            return "gemini_1_5_pro"
        elif api_name == "google/gemma-2-27b-it":
            return "gemma_2_27"
        elif api_name == "gpt-4o-2024-08-06":
            return "gpt_4o"
        elif api_name == "Meta-Llama-3-1-70B-Instruct-ofoo":
            return "Llama_3_70"
        elif api_name == "Meta-Llama-3-1-405B-Instruct":
            return "Llama_3_405"
        elif api_name == "mistral-large-latest":
            return "mistral_large_2"
        elif api_name == "Phi-3-medium-4k-instruct":
            return "Phi_3"
        elif api_name == "Qwen/Qwen2.5-72B-Instruct":
            return "qwen2_5_72"
        elif api_name == "qwen2.5-coder-7b-instruct":
            return "qwen2_5_coder_7"
        elif api_name == "Qwen/Qwen2.5-Coder-32B-Instruct":
            return "qwen2_5_coder_32"
        elif api_name == "starcoder2-15b-instruct-v0.1":
            return "starcoder2_15b_instruct_v0_1" 
        else:
            chars = ["-", ".", ":", "/"]
            for char in chars:
                api_name = api_name.replace(char, "_")
            return api_name


    def run_pytest(self, pytest_params, pytest_queue):
        tr = TestResults()
        try:
            pytest.main(pytest_params, plugins=[tr])
            pytest_queue.put((tr.failed, None))
        except Exception as e:
            pytest_queue.put((None, str(e)))

                
    def trim_answer(self, s):
        if "python" in s:
            s = s.replace("python", "")
        if "Python" in s:
            s = s.replace("Python", "")
        first_backtiks_idx = s.find("```")
        s = s[first_backtiks_idx + 3 :]
        if "```" in s:
            s = s[0 : s.rfind("```")]

        return self.delete_redundant(s)


    def handle_timeout(self, signum, frame):
        raise Exception("Timeout occurred.")


    def set_timeout(self, seconds):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(seconds)

    
    def setup_logging(self, log_file_path):
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )


    def write_info_file(self, q_no, api_name, c, params, function_name, no):
        d = {"params": params, "function_name": function_name}
        with open(f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/info.json", "w") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    def write_info_file(self, q_no, api_name, c, params, function_name, no):
        import os
        import json

        # Ensure folder exists before writing
        folder_dir = os.path.join(f"Q{q_no}", f"{api_name}_results_{no}", f"Folder_{c}")
        os.makedirs(folder_dir, exist_ok=True)

        file_path = os.path.join(folder_dir, "info.json")

        info = {
            "question_number": q_no,
            "model": api_name,
            "folder_number": c,
            "round_number": no,
            "function_name": function_name,
            "parameters": params,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def write_param_file(self, q, model, last_number, params_list):
        length = len(params_list)
        with open(f"Q{q}/{model}_results_{last_number}/all_params.txt", "w") as f:
            for i in range(length):
                if i != length - 1:
                    f.write(f"{str(params_list[i])}\n")
                else:
                    f.write(str(params_list[i]))
    
    def write_param_file(self, q, model, last_number, params_list):
        import os

        # Ensure output directory exists (Windows-safe)
        out_dir = os.path.join(f"Q{q}", f"{model}_results_{last_number}")
        os.makedirs(out_dir, exist_ok=True)

        file_path = os.path.join(out_dir, "all_params.txt")

        length = len(params_list)
        with open(file_path, "w", encoding="utf-8") as f:
            for i in range(length):
                if i != length - 1:
                    f.write(f"{str(params_list[i])}\n")
                else:
                    f.write(str(params_list[i]))


    def write_prompt_and_model_args(self, q_no, c, model, models_params, last_number):
        with open(
            f"Q{q_no}/{model}_results_{last_number}/Folder_{c}/model_specs_prompt.json", "w"
        ) as f:
            json.dump(models_params, f, ensure_ascii=False, indent=2)


    def write_response(self, q_no, c, api_name, answer, no, extension):
        with open(
            f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/response.{extension}", "w"
        ) as f:
            f.write(str(answer))


    def write_test_cases_json_report(self, q_no, api_name, c, params, no):
        with open(f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/test_report.json", "r") as f:
            data = f.read()
        
        results = self.extract_info_from_pytest(data)
        failed_tests = []
        report_dict = {"parameter(s)": params}
        for k, v in results.items():
            if v[0] == "failed":
                failed_tests.append((k, v))
            else:
                report_dict[k] = v
        
        for failed_t in failed_tests:
            report_dict[failed_t[0]] = failed_t[1]
                
        with open(f"Q{q_no}/{api_name}_results_{no}/Folder_{c}/pre_existing_test_case_report.json", "w") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        

    def wrong_params_failure_reasons_writer(self, q_no, model, wrong_params, last_number):
        with open(
            f"Q{q_no}/{model}_results_{last_number}/wrong_params_failure_reasons.json",
            "w",
        ) as f:
            json.dump(wrong_params, f, ensure_ascii=False, indent=2)

    
    def write_html_report(self, path, model_params, seed, q, questions_req, rounds, binary_correctness, number_of_parameters, test_durations):
        version = sys.version_info
        os_type = platform.platform()
        model = model_params.get("model")
        max_tokens = model_params.get("max_tokens")
        temperature = model_params.get("temperature")
        model_path = self.rename_model(model)
        
        report_content = f"""
        <table style="width:50%" class="question">
        <tr>
            <td>Question {q}</td>
            <td></td>
        </tr>
        <tr>
            <td>Number of parameters</td>
            <td>{questions_req[q].get('number_of_parameters')}</td>
        </tr>
        <tr>
            <td>Parameters shuffle</td>
            <td>{questions_req[q].get('shuffle')}</td>
        </tr>
        <tr>
            <td>Fuzzy testing</td>
            <td>{questions_req[q].get('fuzzy_testing')}</td>
        </tr>
        <tr>
            <td>Number of fuzzy inputs</td>
            <td>{questions_req[q].get('number_of_fuzzy_inputs')}</td>
        </tr>
        """
        for i in range(len(rounds)):
            corr_round = sum(binary_correctness[i*number_of_parameters:(i+1)*number_of_parameters])
            link_wrong_params = (
            f'<a href="{model_path}_results_{rounds[i]}/wrong_params.html" target="_blank">Click to see them.</a>'
            if number_of_parameters - corr_round
            else ""
            )
            link_correct_params = (
            f'<a href="{model_path}_results_{rounds[i]}/correct_params.html" target="_blank">Click to see them.</a>'
            if corr_round
            else ""
            )

            report_content += f"""
            <tr>
                <td>Results of Round {rounds[i]}</td>
                <td></td>
            </tr>
            <tr>
                <td>Number of correct responses</td>
                <td>{corr_round}&nbsp&nbsp&nbsp&nbsp&nbsp{link_correct_params}</td>
            </tr>
            <tr>
                <td>Number of wrong responses</td>
                <td>{number_of_parameters - corr_round}&nbsp&nbsp&nbsp&nbsp&nbsp{link_wrong_params}</td>
            </tr>
            <tr style="border-bottom: 3px solid black;">
                <td>Test duration</td>
                <td>{round(test_durations[i], 3)} (s)</td>
            </tr>
            <br></br>
            """

        content = (
            f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" href="../report_styles.css">
        </head>
        <body>
            <h1>Turbulence Benchmark</h1>
            <table style="width:100%" class="info_table">
                <tr>
                    <td>Operating system</td>
                    <td>{os_type}</td>
                </tr> 
                <tr>
                    <td>Python version</td>
                    <td>{version[0]}.{version[1]}.{version[2]}</td>
                </tr>  
                <tr>
                    <td>Large language model</td>
                    <td>{model} (max_tokens: {max_tokens}, temperature: {temperature})</td>
                </tr>
                <tr>
                <td>Seed value</td>
                <td>{seed}</td>
                </tr>
            </table>"""
            + report_content
            + """
        </body>
        </html>
        """
        )

        full_path = path + f"/Q{q}"
        with open(
            full_path + f'/Final_report_{time.strftime("%Y%m%d-%H%M%S")}.html', "w"
        ) as f:
            f.write(content)
