import hashlib
import json
from time import time
from uuid import uuid4
from flask import Flask, jsonify, request
from werkzeug.security import generate_password_hash, check_password_hash
from Crypto.Cipher import AES
import base64
import os

class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_votes = []
        self.voters = {}
        self.encryption_key = os.urandom(32)  # Generate a random encryption key
        self.new_block(previous_hash='1', proof=100)

    def new_block(self, proof, previous_hash=None):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'votes': self.current_votes,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.current_votes = []
        self.chain.append(block)
        return block

    def register_voter(self, email, password):
        if email in self.voters:
            raise ValueError("Voter already registered")
        hashed_password = generate_password_hash(password)
        self.voters[email] = hashed_password
        return email

    def authenticate_voter(self, email, password):
        if email in self.voters and check_password_hash(self.voters[email], password):
            return True
        return False

    def encrypt_vote(self, vote):
        cipher = AES.new(self.encryption_key, AES.MODE_EAX)
        ciphertext, tag = cipher.encrypt_and_digest(json.dumps(vote).encode())
        return base64.b64encode(cipher.nonce + tag + ciphertext).decode()

    def new_vote(self, voter_id, candidate):
        encrypted_vote = self.encrypt_vote({'voter_id': voter_id, 'candidate': candidate})
        self.current_votes.append(encrypted_vote)
        return self.last_block['index'] + 1

    @staticmethod
    def hash(block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        return self.chain[-1]

    def proof_of_work(self, last_proof):
        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof, proof):
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

app = Flask(__name__)
node_identifier = str(uuid4()).replace('-', '')

blockchain = Blockchain()

@app.route('/mine', methods=['GET'])
def mine():
    last_block = blockchain.last_block
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)

    previous_hash = blockchain.hash(last_block)
    block = blockchain.new_block(proof, previous_hash)

    response = {
        'message': "New Block Forged",
        'index': block['index'],
        'votes': block['votes'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200

@app.route('/votes/new', methods=['POST'])
def new_vote():
    values = request.get_json()
    required = ['voter_id', 'candidate']
    if not all(k in values for k in required):
        return 'Missing values', 400

    index = blockchain.new_vote(values['voter_id'], values['candidate'])
    response = {'message': f'Vote will be added to Block {index}'}
    return jsonify(response), 201

@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain),
    }
    return jsonify(response), 200

@app.route('/voters/register', methods=['POST'])
def register_voter():
    values = request.get_json()
    email = values.get('email')
    password = values.get('password')
    try:
        voter_id = blockchain.register_voter(email, password)
        response = {'message': f'Voter {voter_id} registered successfully'}
        return jsonify(response), 201
    except ValueError as e:
        response = {'message': str(e)}
        return jsonify(response), 400

@app.route('/voters/authenticate', methods=['POST'])
def authenticate_voter():
    values = request.get_json()
    email = values.get('email')
    password = values.get('password')
    if blockchain.authenticate_voter(email, password):
        response = {'message': 'Authentication successful'}
        return jsonify(response), 200
    else:
        response = {'message': 'Authentication failed'}
        return jsonify(response), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
