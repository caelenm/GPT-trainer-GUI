**Instructions**

*(Create and run in a venv if you want to separate dependencies)*

```bash
cd GPT-trainer-GUI-main
pip install -r requirements.txt
python GUI.py
```


Executive summary: 
Is a desktop application written in python that focuses on providing a simple interface for the fine-tuning and experimentation with open-source large language models. The project was inspired by the lack of simple, generalized pytorch or TensorFlow implementations and limited beginner-friendly educational software products that address this problem. Initial plans looked at creating an application that fine-tunes an LLM on user provided documents, but that plan was discarded due to limited value-add and high user friction. Functional requirements focus on providing a simple interface that lets the user choose a model, dataset, and parameters that suit their needs. Non-functional requirements aim to make the application fast, responsive, and cross-platform. The software consists of independent modules that interact through message passing, leading to high cohesion and low coupling. The project is intended for educational and research purposes and can be extended or connected with other similar applications. It allows for experimentation with GPT2 variants and Taskmaster2 datasets.  In-depth manual testing was done to ensure a good UX, useful features, and satisfactory model responses. Several of the most popular machine learning and application development tools were used such as pytorch, Nvidia CUDA, and Tkinter. Future work intends to improve UI, expand model options, and allow for more complex interfaces for advanced users. Overall, the project is a standalone and useful tool for creating and interacting with large language models. 
