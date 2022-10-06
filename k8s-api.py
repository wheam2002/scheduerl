from kubernetes import client, config, watch
import json


config.load_kube_config()
k8sCoreV1api = client.CoreV1Api()
scheduler_name = 'test'


def get_node():
    nodeInstance = []
    nodeInstanceList = k8sCoreV1api.list_node()
    for i in nodeInstanceList.items:
        nodeInstance.append(i.metadata.name)
    return nodeInstance

def get_pending_pod():
    pending_pod = []
    w = watch.Watch()
    for event in w.stream(k8sCoreV1api.list_namespaced_pod, "default"):
        # and event['object'].spec.scheduler_name == scheduler_name:
        if event['object'].status.phase == "Pending":
            try:
                print(event['object'].metadata.name)
                # object_now = json.loads(event['object'].metadata.annotations['com.openfaas.function.spec'])
                # type(object_now['limits'])
                print(event['object'].metadata.limits)
                
                # res = scheduler(event['object'].metadata.name,random.choice(nodes_available()))
            except client.rest.ApiException as e:
                print(json.load(e.body)["message"])

def main():
    # api_response = k8sCoreV1api.read_node('kaip-3', pretty=True)
    # print(api_response)
    w = watch.Watch()
    for event in w.stream(k8sCoreV1api.list_namespaced_pod, "openfaas-fn"):
        # and event['object'].spec.scheduler_name == scheduler_name:
        if event['object'].status.phase == "Pending":
            try:
                # print(event['object'].metadata.name)
                # fdfsdf = afas
                object_now = json.loads(event['object'].metadata.annotations['com.openfaas.function.spec'])
                print(object_now)
                # type(object_now['limits'])
                # print(event['object'].metadata.limits)
                # res = scheduler(event['object'].metadata.name,random.choice(nodes_available()))
            except client.rest.ApiException as e:
                print('1111111')
                print(json.load(e.body)["message"])


if __name__ == '__main__':
    nodeInstance = []
    nodeInstanceList = k8sCoreV1api.list_node()
    for i in nodeInstanceList.items:
        print(i)
        nodeInstance.append(i.metadata.name)
    # print(nodeInstanceList)