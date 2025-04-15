from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, re, time
from werkzeug.utils import secure_filename
from datetime import datetime
from vectorstore import response_to_prompt, response_with_fix, customer_biodata_to_pinecone, customer_integrations_to_pinecone, app_risk_to_pinecone, customer_app_to_pinecone, admin_update_vector_store, structured_update_vector_store, unstructured_update_vector_store, stringcontent_update_vector_store, codefile_update_vector_store, pinecone_namespace_delete_repo
#import testdata

app = Flask(__name__)
#CORS(app, resources={r"/*": {"origins": "https://platform-staging.rezliant.com"}})
#CORS(app, resources={r"/*": {"origins": "*"}})

CORS(app, resources={r"/*": {"origins": [
    "https://app.rezliant.com",
    "https://stg.rezliant.com",
    "https://developer.rezliant.com",
    "https://reposcanner.developer.rezliant.com",
    "https://reposcanner.stg.rezliant.com"
    "https://platform-staging.rezliant.com",
    "https://dashboard-api.stg.rezliant.com",
    "https://dashboard-api.dev.rezliant.com"
    "https://gitlab-468l.onrender.com",
    "http://localhost:3000"
]}})

conversation = []
current_model = "amazon.nova-micro-v1:0" #
#anthropic.claude-3-5-sonnet-20240620-v1:0
#amazon.nova-lite-v1:0
#amazon.nova-micro-v1:0
#amazon.nova-pro-v1:0

app.config['DOCUMENT_FOLDER'] = 'data'
allowed_extensions = {'pdf', 'txt', 'docx', 'html'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def home():
    return send_from_directory('templates', 'index.html')

@app.route('/set_model', methods=['POST'])
def set_model():
    global current_model
    data = request.json
    current_model = data["model_name"]
    files = os.listdir(app.config['DOCUMENT_FOLDER'])
    return jsonify(files=files, model=current_model)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data["user_id"]
    user_persona = data["user_persona"]
    user_input = data["user_input"]
    moment = "chat"
    llm_response = response_to_prompt(moment, user_id, user_persona, user_input, current_model)
    #files = os.listdir(app.config['DOCUMENT_FOLDER'])
    return jsonify(user_input=user_input, llm_response=llm_response)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200
    
    
@app.route('/fix', methods=['POST'])
def fix():
    data = request.json
    user_id = data["user_id"]
    user_input = data["user_input"] + ".\nWhat do you recommend?" #do this append in final function along with prompt
    user_persona = data["user_persona"]
    moment = "fix"
    llm_response = response_with_fix(moment, user_id, user_persona, user_input, current_model)
    #files = os.listdir(app.config['DOCUMENT_FOLDER'])
    return jsonify(user_persona=user_persona, user_input=user_input, llm_response=llm_response)


@app.route('/fixdev', methods=['POST'])
def fixdev():
    data = request.json
    print(data)
    user_id = data["user_id"]
    user_input = data["user_input"]
    user_persona = data["user_persona"]

    #Extraneous details about vulnerability
    file_name = "\nFile Name: " + data["file_name"] + "\n"
    code_snippet = "Code Snippet: " + data["code_snippet"] + "\n"
    references = "References: " + data["references"] + "\n"
    owasp = "Owasp: " + data["owasp"] + "\n"

    vuln_details = file_name +code_snippet + references + owasp + ".\nGive me one recommended fix?"
    user_input = user_input + vuln_details

    moment = "fix"
    llm_response = response_with_fix(moment, user_id, user_persona, user_input, current_model)
    #files = os.listdir(app.config['DOCUMENT_FOLDER'])
    return jsonify(user_persona=user_persona, user_input=user_input, llm_response=llm_response)


#user biodata
@app.route('/embed_user_biodata', methods=['POST'])
def embed_user_biodata():
    data = request.json

    userId = data["userId"]
    companyInfo = data["companyInfo"]
    companySize = data["companySize"]
    position = data["position"]
    industry = data["industry"]

    embed_response = customer_biodata_to_pinecone(userId, companyInfo, companySize, position, industry)
    #files = os.listdir(app.config['DOCUMENT_FOLDER'])
    return jsonify(embed_response=embed_response)


#user integration data
@app.route('/embed_user_integrations', methods=['POST'])
def embed_user_integrations():
    data = request.json
    integrations = data["integrations"]
    embed_response = customer_integrations_to_pinecone(integrations)
    #files = os.listdir(app.config['DOCUMENT_FOLDER'])
    return jsonify(embed_response=embed_response)


#user application risk
@app.route('/embed_app_risk', methods=['POST'])
def embed_app_risk():
    data = request.json
    risk = data["risk"]
    embed_response = app_risk_to_pinecone(risk)
    #files = os.listdir(app.config['DOCUMENT_FOLDER'])
    return jsonify(embed_response=embed_response)


#user application data
@app.route('/embed_user_details', methods=['POST'])
def embed_user_details():
    data = request.json
    user_id = data["user_id"]
    app_name = data["application_name"]
    app_description = data["application_description"]
    tech_stack = data["tech_stack"]
    embed_response = customer_app_to_pinecone(user_id, app_name, app_description, tech_stack)
    #files = os.listdir(app.config['DOCUMENT_FOLDER'])
    return jsonify(embed_response=embed_response)


#list uploaded files on server
@app.route('/get_files', methods=['GET'])
def get_files():
    files = os.listdir(app.config['DOCUMENT_FOLDER'])
    return jsonify(files=files)


@app.route('/admin_update_vector_store', methods=['POST'])
def admin_update_store():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    files = request.files.getlist('file')

    for file in files:
        if file.filename == '':
            return jsonify(error="No selected file"), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['DOCUMENT_FOLDER'], filename)
            file.save(file_path)
            #admin_update_vector_store(file_path, "rezliant", "rezdocuments")
            structured_update_vector_store("94b8a448-00a1-704b-f8d8-3452bbb61092", file_path, filename, "The dependencies needed to run the app")
            #make sure to delete file from temp directory

        if file and not allowed_file(file.filename): #file extension isn't supported
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['DOCUMENT_FOLDER'], filename)
            file.save(file_path)
            unstructured_update_vector_store("94b8a448-00a1-704b-f8d8-3452bbb61092", file_path, filename, "We ran Semgrep on the app and this was the scan result")
            #make sure to delete file from temp directory

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vector_updated_message = f"Vector Store Updated at {timestamp}"
    files = os.listdir(app.config['DOCUMENT_FOLDER'])
    return jsonify(vector_updated_message=vector_updated_message, files=files)


@app.route('/user_update_vector_store', methods=['POST'])
def user_update_store():
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    files = request.files.getlist('file')

    for file in files:
        if file.filename == '':
            return jsonify(error="No selected file"), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['DOCUMENT_FOLDER'], filename)
            file.save(file_path) #save file to temp directory
            #structured_update_vector_store(file_path, "rezliant", "rezdocuments") #parameters - userid, filepath, description
            structured_update_vector_store(user_id, file_path, filename, file_description)
            #make sure to delete file from temp directory

        if file and not allowed_file(file.filename): #file extension isn't supported
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['DOCUMENT_FOLDER'], filename)
            file.save(file_path)
            unstructured_update_vector_store(user_id, file_path, filename, file_description)
            #make sure to delete file from temp directory

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vector_updated_message = f"Vector Store Updated at {timestamp}"
    files = os.listdir(app.config['DOCUMENT_FOLDER'])
    return jsonify(vector_updated_message=vector_updated_message, files=files)


@app.route('/rag_vulnerable_file', methods=['POST'])
def rag_vulnerable_file():
    data = request.json
    print("***********************************")
    print(data)
    user_id = data["user_id"]
    file_content = data["file"]
    repo_name = data["reponame"]
    absolute_path = data["filename"] #contains absolute path/ aka full working directory

    #splitting the path
    directory, file_name = os.path.split(absolute_path)
    id_prefix = repo_name + "#" + directory + "#" + file_name

    codefile_update_vector_store(user_id, file_content, file_name, id_prefix, None)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vector_updated_message = f"Vector Store Updated at {timestamp}"
    return jsonify(vector_updated_message=vector_updated_message)


@app.route('/rag_vulnerable_files_from_array', methods=['POST'])
def rag_vulnerable_files_from_array():
    data = request.json
    if not isinstance(data, list):  # Ensure the input is a list
        return jsonify(error="Invalid input: expected a list of dictionaries"), 400

    results = []
    print("***********************************")
    #print(data)
    #try to print number of items in array, this will represent the number of vulerabilities

    # for each file, rag the content of that file
    for item in data:
        try:
            user_id = item["user_id"]
            file_content = item["file"] #contains code
            repo_name = item["reponame"]
            absolute_path = item["filename"] #contains full working directory

            #splitting the path
            directory, file_name = os.path.split(absolute_path)
            stripped_directory = re.sub(r".*/repo_\d{8}_\d{6}", "", directory)
            #id_prefix = repo_name + "#" + directory + "#" + file_name
            id_prefix = repo_name + directory + "/" + file_name
            description = "Record ID is in the format: AbsolutePath_ChunkNumber"

            #delete repo chunks if they exist
            pinecone_namespace_delete_repo("customer" + user_id, repo_name) #args - namespace, id_prefix

            #rag code to pinecone
            codefile_update_vector_store(user_id, file_content, file_name, id_prefix, description)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vector_updated_message = f"Vector Store Updated at {timestamp} for file: {file_name}"
            results.append({"file_name": file_name, "message": vector_updated_message})

        except KeyError as e:
            # Handle missing keys in the dictionary
            results.append({"error": f"Missing key {e.args[0]} in item: {item}"})

    print("***********************************")
    print(results)
    return jsonify(results=results)


@app.route('/delete_repo', methods=['POST'])
def delete_repo():
    data = request.json
    user_id = data["user_id"]
    repo_name = data["reponame"]

    #delete repo chunks if they exist
    pinecone_namespace_delete_repo("customer" + user_id, repo_name) #args - namespace, id_prefix

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    vector_updated_message = f"Vector Store Updated at {timestamp}"
    return jsonify(vector_updated_message=vector_updated_message)


def extract_list(llm_response):
    # Use regex to find the list in the string
    #match = re.search(r'\[(.*?)\]', output)
    match = re.search(r'\[(.*?)\]', llm_response)
    if match:
        # Extract the content within the brackets and convert to a list of integers
        return [int(x) for x in match.group(1).split(',')]
    return []  # Return an empty list if no list is found


@app.route('/vulnerability_reranker', methods=['POST'])
def vulnerability_reranker():
    data = request.json
    #print(data)

    # Extracting values
    findings = data['findings']
    repository = data['metadata']['repository']
    user_id = data['metadata']['user_id']

    #print("Findings:", findings)
    #print("Repository:", repository)
    #print("User ID:", user_id)

    moment = "scan"
    user_persona = "ranker"
    user_input = (
        "Rank these vulnerabilities from most critical to lowest risk, return the rankings as a list of numbers which are representative of the ID value of each vulnerability object in these findings array: \n"
        + str(findings)
        + "\nRepository/App name: "
        + repository
    )
    llm_response = response_to_prompt(moment, user_id, user_persona, user_input, current_model)
    llm_response = extract_list(llm_response)

    return jsonify(llm_response=llm_response)

@app.route('/vulnerability_reranker2', methods=['POST'])
def vulnerability_reranker2():
    data = testdata.llm_data

    # Extracting values
    findings = data['findings']
    #print(findings)
    repository = "vulnerable-code-snippets"
    user_id = "6428f418-d0e1-7044-155b-a3e4c265ec81"

    #print("Findings:", findings)
    #print("Repository:", repository)
    #print("User ID:", user_id)

    moment = "scan"
    user_persona = "ranker"
    user_input = (
        "Rank these vulnerabilities from most critical to lowest risk, return the rankings as a list of numbers which are representative of the ID value of each vulnerability object in these findings array: \n"

        + str(findings)
        + "\nRepository/App name: "
        + repository
    )
    llm_response = response_to_prompt(moment, user_id, user_persona, user_input, current_model)
    llm_response = extract_list(llm_response)
    #print(llm_response)

    #return jsonify(llm_response=llm_response)

@app.route('/delete_file/<filename>', methods=['DELETE','POST'])
def delete_file(filename):
    file_path = os.path.join(app.config['DOCUMENT_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    files = os.listdir(app.config['DOCUMENT_FOLDER'])
    return jsonify(files=files)

#Josh updated
@app.route('/cloud_vulnerability_reranker', methods=['POST'])
def cloud_vulnerability_reranker():
    data = request.json

    # Extracting values
    findings = data['findings']
    category = data.get('metadata', {}).get('category', 'Unknown')
    user_id = data.get('metadata', {}).get('user_id', 'Unknown')

    moment = "cloudscan"
    user_persona = "ranker"
    user_input = (
        "Rank these cloud security vulnerabilities from most critical to lowest risk, return the rankings as a list of numbers which are representative of the ID value of each vulnerability object in these findings array: \n"
        + str(findings)
        + "\nCategory: "
        + category
    )
    llm_response = response_to_prompt(moment, user_id, user_persona, user_input, current_model)
    llm_response = extract_list(llm_response)

    return jsonify(llm_response=llm_response)


if __name__ == '__main__':
    os.makedirs(app.config['DOCUMENT_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=2024, debug=True)