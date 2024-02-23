import requests

def retrieve(query):
    """
    Sends a query to the Flask server to retrieve information from vectorized documents.
    
    Args:
        query (str): The query string to send for document retrieval.
    
    Returns:
        string: Context retrieved based on the query.
    """
    # Endpoint URL of the Flask server
    url = "http://127.0.0.1:5000/rag"
    
    # Prepare the request payload
    payload = {"query": query}
    
    # Send the POST request
    try:
        response = requests.post(url, json=payload)
        data = response.json()  # Parse the JSON response
        return data["document"]
    
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while retrieving documents: {e}")
        return ""

# Example usage
if __name__ == "__main__":
    query = "Ask any question here."
    documents = retrieve(query)
    print(documents)
