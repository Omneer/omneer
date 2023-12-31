<div align="center">

# Omneer SDK

Omneer SDK is a state-of-the-art toolkit designed for the development and deployment of AI and machine learning powered personalized medicine applications in the neuroscience field. The SDK, with its Python-centric approach, provides an easy-to-use platform for developers to create and interact with their applications.

Fueled by Airflow, a dynamic and highly scalable workflow orchestration engine known for its efficiency and flexibility, Omneer SDK ensures seamless processing and management of complex data workflows. Airflow's Directed Acyclic Graph (DAG) approach ensures atomicity at a task level, allowing for a comprehensive and orderly execution of tasks.

This SDK further enhances workflow execution with containerization options, offering independence in task scheduling and a vast array of scalable computing resources. As such, the Omneer SDK represents a perfect convergence of neuroscience, personalized medicine, and cutting-edge technology, promising a new era of advanced and personalized healthcare solutions.

[Docs](https://docs.omneer.xyz) • [Installation](#installation) •
[Quickstart](#configuration) • [Omneer](https://omneer.xyz)

</div>

Utilizing the Omneer SDK, developers can deliver:

- Personalized AI and machine learning tools
- Advanced disease progression diagnosis and tracking
- Creation of digital twins for individualized healthcare
- Instant no-code interfaces for rapid deployment
- High-performing, reliable cloud infrastructure
- Flexibility to define resources (CPU, GPU, etc.) for serverless execution

- Omneer SDK continues to be a ground-breaking platform for personalized healthcare solutions. Browse our collection of existing and actively maintained solutions at [Omneer Community]().

### Getting Started

See the SDK in action by following the steps below to register your first workflow with Omneer.

First, install omneer through `pip`.

```
$ pip install omneer
```

Then, create some boilerplate code for your new workflow.

```
$ omneer init diagnosis
```

The registration process, which could take a few minutes depending on your network connection, involves building a Docker image with your workflow code, serializing the code, registering it with your Omneer account, and pushing the Docker image to a managed container registry.

Upon successful registration, your new workflow should be visible in your Omneer Console.

For issues with registration or other queries, please raise an issue on GitHub.

---

### Installation

Omneer SDK is distributed via pip. We recommend installing it in a clean virtual environment for the best user experience.

Virtualenv is our recommended tool for creating isolated Python environments.

[Virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) is recommended.

```
pip install omneer
```

### Examples

[Omneer Examples]() features list of well-curated workflows developed by the Omneer team. 
* [Parkinson's Diagnosis]()
* [Parkinson's Progression]()
* [Parkinson's Personalized]()

We'll maintain a growing list of well documented examples developed by our community members here. Please open a pull request to feature your own:

**Parkinson's Diagnosis**
  * [Early Detection]()

 
  
