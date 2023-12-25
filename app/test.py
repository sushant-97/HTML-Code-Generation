import requests
import argparse

def make_post_request(url, form_data):
    try:
        response = requests.post(url, data=form_data)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print("Request was successful!")
            print("Response data:", response.text)
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response data:", response.text)
            
    except requests.RequestException as e:
        print(f"Error making the request: {e}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Make a POST request to an API')
    parser.add_argument('--api_url', type=str, required=False, help='API endpoint URL')
    parser.add_argument('--user_prompt', type=str, required=True, help='User prompt for the API request')
    

    args = parser.parse_args()

    # Replace 'your_api_endpoint' with the actual API endpoint URL
    # api_url = 'http://192.168.2.119:8888/query'
    api_url_ = args.api_url

    # Use the user-provided prompt as form data
    form_data = {
        'user_prompt': args.user_prompt
    }

    # Make the POST request
    make_post_request(api_url_, form_data)

if __name__ == "__main__":
    main()