# %%
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector, random_statevector
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import QasmSimulator
import numpy as np
# import NumPy for linear algebra computations
import numpy as np
from numpy.linalg import matrix_power 

# import random for generating random numbers
import random

# import copy for copying lists
import copy

# %% [markdown]
# ## Quantum state creation
# 

# %%
#define simulator 
simulator = QasmSimulator()

def gen_qubit(alpha, beta):
    """
    Generate a single qubit according to the probability amplitudes alpha and beta and the constraint alpha^2 + beta^2 = 1.
    """
    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)
    init_gate = Initialize([alpha, beta])  # Wrap in a list
    qc.append(init_gate, [qr[0]])
    qc = transpile(qc, simulator)
    return qc

def zero_state():
    """
    Generate the zero state |0>.
    
    """
    return gen_qubit(1,0)

def one_state():
    """
    Generate the one state |1>.

    """
    
    return gen_qubit(0,1)
def superposition():
    """
    Generate the plus state |+> = 1/sqrt(2) * (|0> + |1>).
    
    Returns:
        - superposition_circuit: A Qiskit circuit with the |+> state prepared.
    """
    qr = QuantumRegister(1)
    superposition_circuit = QuantumCircuit(qr)
    superposition_circuit.h(qr[0])  # Apply Hadamard gate to create superposition
    superposition_circuit = transpile(superposition_circuit, simulator)
    return superposition_circuit 
def gen_epr():
    """
    Generate the EPR state 1/sqrt(2) * (|00> + |11>).
    
   Returns:
        - epr_circuit: A Qiskit circuit with the EPR state prepared.
    """
    qr = QuantumRegister(2)
    epr_circuit = QuantumCircuit(qr)
    epr_circuit.h(qr[0])  # Hadamard gate on the first qubit
    epr_circuit.cx(qr[0], qr[1])  # CNOT with control on qubit 0 and target on qubit 1
    epr_circuit = transpile(epr_circuit, simulator)
    return epr_circuit
def computational_basis(n: int):
    """
    Generate all the basis elements of an n-qubit system in the computational basis
    as Qiskit Statevector objects.
    
    Args:
        - n: the number of qubits
        
    Returns:
        - basis_states: a list of Statevector objects representing the computational basis states.
    """
    basis_states = []
    for i in range(2**n):
        # Create a binary string for each basis state (e.g., "00", "01" for n=2)
        binary_string = bin(i)[2:].zfill(n)
        
        # Create a QuantumCircuit to prepare the basis state
        qc = QuantumCircuit(n)
        for qubit, bit in enumerate(binary_string):
            if bit == '1':
                qc.x(qubit)  # Apply X gate to flip to |1> if the binary bit is 1

        # Convert the circuit to a Statevector and add to the list
        basis_states.append(Statevector.from_instruction(qc))
    
    return basis_states
def measurement(qc):
    """
    Measure the quantum circuit in the computational basis and return the result as a binary string.

    Args:
        - qc: QuantumCircuit instance to be measured.
        
    Returns:
        - classical_bits: Measurement result in binary string format.
    """
    # Get the number of qubits and add a classical register for measurements
    num_qubits = qc.num_qubits
    cr = ClassicalRegister(num_qubits)
    qc.add_register(cr)
    
    # Measure all qubits in the circuit
    qc.measure(qc.qregs[0], cr)
    
    # Execute the circuit on the QasmSimulator
    job = simulator.run(transpile(qc, simulator), shots=1)
    result = job.result()
    
    # Get the measurement result as a dictionary and extract the bitstring
    counts = result.get_counts(qc)
    classical_bits = list(counts.keys())[0]  # Get the measurement result as a binary string
    
    return classical_bits

# %%
def post_measurement_state(qc, L, bits):
    """
    Given a QuantumCircuit, indices of subsystems to measure, and a desired outcome,
    this function returns the post-measurement state of the unmeasured subsystem(s) 
    conditioned on observing the specified measurement outcome.

    Args:
        - qc: QuantumCircuit to be measured.
        - L: List of qubit indices to measure.
        - bits: String of expected measurement outcomes for the measured qubits in L.

    Returns:
        - post_measurement_density: Density matrix of the post-measurement state.
        - bits: The measurement outcomes as provided.
    """
    # Create a copy of the input circuit to add measurement
    num_qubits = qc.num_qubits
    measured_circuit = qc.copy()
    
    # Add classical register for measurement results
    cr = ClassicalRegister(len(L))
    measured_circuit.add_register(cr)
    
    # Measure the specified qubits into the classical register
    for idx, qubit in enumerate(L):
        measured_circuit.measure(qubit, cr[idx])

    # Execute the circuit
    job = simulator.run(measured_circuit, shots=1024)
    result = job.result()
    counts = result.get_counts(measured_circuit)

    # Check if the desired outcome `bits` was observed
    if bits not in counts:
        raise ValueError(f"Outcome '{bits}' was not observed in the measurement results.")

    # Calculate the probability of observing `bits`
    probability = counts[bits] / sum(counts.values())

    # Get the density matrix of the initial state
    rho = DensityMatrix.from_instruction(qc)
    
    # Evolve the density matrix by the measurement circuit
    post_rho = rho.evolve(measured_circuit)

    # Extract the state of the unmeasured qubits
    unmeasured_qubits = list(set(range(num_qubits)) - set(L))
    post_measurement_density = partial_trace(post_rho, unmeasured_qubits)

    # Normalize by probability to represent the conditional post-measurement state
    post_measurement_density = post_measurement_density / probability

    return post_measurement_density, bits

# %% [markdown]
# ## Quantum One-Time Pad (QOTP) Encryption and Decryption
# 

# %%
def qotp_enc(rho_enc, qubits, a, b):
    """
    Applies Quantum One-Time Pad (QOTP) encryption on a DensityMatrix.
    
    Args:
        - rho_enc: DensityMatrix to encrypt.
        - qubits: List of qubits in the circuit to apply the encryption to.
        - a: List of bits for the X gates in the QOTP encryption.
        - b: List of bits for the Z gates in the QOTP encryption.
        
    Returns:
        - rho_enc: The encrypted DensityMatrix object.
    """
    # Ensure rho_enc is a DensityMatrix
    if isinstance(rho_enc, QuantumCircuit):
        rho_enc = DensityMatrix.from_instruction(rho_enc)

    for i, qubit in enumerate(qubits):
        if b[i] == 1:
            z_circuit = QuantumCircuit(rho_enc.num_qubits)
            z_circuit.z(i)  # Use index `i` directly
            rho_enc = rho_enc.evolve(z_circuit)
        if a[i] == 1:
            x_circuit = QuantumCircuit(rho_enc.num_qubits)
            x_circuit.x(i)  # Use index `i` directly
            rho_enc = rho_enc.evolve(x_circuit)
    return rho_enc

def qotp_dec(rho_dec_eval, qubits, a, b):
    """
    Apply Quantum One-Time Pad (QOTP) decryption on a DensityMatrix.
    
    Args:
        - rho_dec_eval: DensityMatrix to decrypt.
        - qubits: List of qubits in the circuit to apply the decryption to.
        - a: List of bits for the X gates in the QOTP decryption.
        - b: List of bits for the Z gates in the QOTP decryption.
        
    Returns:
        - rho_dec_eval: The decrypted DensityMatrix object.
    """
    for i, qubit in enumerate(qubits):
        if b[i] == 1:
            z_circuit = QuantumCircuit(rho_dec_eval.num_qubits)
            z_circuit.z(qubit)
            rho_dec_eval = rho_dec_eval.evolve(z_circuit)
        if a[i] == 1:
            x_circuit = QuantumCircuit(rho_dec_eval.num_qubits)
            x_circuit.x(qubit)
            rho_dec_eval = rho_dec_eval.evolve(x_circuit)
    return rho_dec_eval

# %% [markdown]
# ## Circuit Decomposition and Analysis

# %%
def no_qubits(circuit):
    """
    Get the number of qubits in the QuantumCircuit.

    Args:
        - circuit: QuantumCircuit object.

    Returns:
        - number_of_qubits: Number of qubits in the circuit.
    """
    return circuit.num_qubits

def number_of_tgates(circuit):
    """
    Count the number of T gates in a QuantumCircuit.

    Args:
        - circuit: QuantumCircuit object.

    Returns:
        - t_gate_count: Total number of T gates in the circuit.
    """
    t_gate_count = sum(1 for instruction in circuit.data if instruction.operation.name == "t")
    return t_gate_count
def wire_dictionary(circuit):
    """
    Construct a dictionary to track actions and T gate applications on each qubit in a QuantumCircuit.

    Args:
        - circuit: QuantumCircuit object.

    Returns:
        - circuit_dictionary: Dictionary where keys are qubit indices (as strings) and values are lists containing:
            [qubit index, T gate applied (True/False), total T gates on qubit, T gates applied so far].
    """
    num_qubits = circuit.num_qubits
    circuit_dictionary = {str(i): [i, False, 0, 0] for i in range(num_qubits)}

    # Loop through each gate in the circuit and update the dictionary
    for gate, qubits, _ in circuit.data:
        for qubit in qubits:
            qubit_index = str(qubit.index)
            if gate.name == "t":
                circuit_dictionary[qubit_index][1] = True  # T gate has been applied
                circuit_dictionary[qubit_index][2] += 1    # Increment total T gate count

    return circuit_dictionary

def no_wires(circuit):
    """
    Calculate the total number of wires required for the EPR evaluation scheme.

    Args:
        - circuit: QuantumCircuit object.

    Returns:
        - number_of_wires: Total number of wires, calculated as number of qubits + 2 * number of T gates.
    """
    number_of_qubits = circuit.num_qubits
    t_gate_count = number_of_tgates(circuit)
    number_of_wires = number_of_qubits + 2 * t_gate_count
    return number_of_wires

def number_of_tgates_dictionary(circuit_dictionary):
    """
    Count the total number of T gates in a circuit from the circuit dictionary.

    Args:
        - circuit_dictionary: Dictionary from `wire_dictionary` function.

    Returns:
        - number_of_t_gates: Total number of T gates in the circuit.
    """
    number_of_t_gates = sum(info[2] for info in circuit_dictionary.values())
    return number_of_t_gates

# %% [markdown]
# ## Layer Circuit Structure and Gate operation

# %%
def tcl_layers(circuit):
    """
    Separate T and Clifford gates in each layer of a QuantumCircuit.

    Args:
        - circuit: QuantumCircuit object.

    Returns:
        - layered_circuit: A nested list where each inner list represents a layer 
          and contains strings indicating the gates in that layer.
    """
    layered_circuit = []
    for instruction, qubits, _ in circuit.data:
        gate_name = instruction.name
        if gate_name in {"cx", "cz", "h", "s", "x", "y", "z", "t", "id"}:  # Common Clifford + T gate names in Qiskit
            layered_circuit.append([gate_name.upper()])  # Qiskit uses lowercase gate names by default

    return layered_circuit
def cl_eval_no_fhe(circuit, enc_psi):
    """
    Perform Clifford circuit evaluation on an encrypted pure state homomorphically using Qiskit.

    Args:
        - circuit: QuantumCircuit containing the Clifford circuit.
        - enc_psi: Statevector representing the encrypted state.

    Returns:
        - rho_hom_eval: DensityMatrix of the homomorphically evaluated quantum state.
        - c_list: A list of measurement outcomes for each qubit.
    """    
    # Initialize density matrix from the encrypted state
    rho_hom_eval = DensityMatrix(enc_psi)

    # Initializing a list to store measurement results
    number_of_qubits = circuit.num_qubits
    c_list = [[] for _ in range(number_of_qubits)]

    # Iterate through the gates, applying Clifford gates to the density matrix
    for instruction, qubits, _ in circuit.data:
        gate_name = instruction.name
        if gate_name in {"h", "x", "z", "s", "id"}:  # Handle single-qubit Clifford gates
            for qubit in qubits:
                rho_hom_eval = rho_hom_eval.evolve(instruction, qubit.index)
        elif gate_name == "cx":  # Handle two-qubit Clifford (e.g., CNOT)
            control_qubit = qubits[0].index
            target_qubit = qubits[1].index
            rho_hom_eval = rho_hom_eval.evolve(instruction, [control_qubit, target_qubit])

    # Generate the circuit dictionary for additional tracking if needed
    circuit_dictionary = wire_dictionary(circuit)
    return rho_hom_eval, c_list

# %% [markdown]
# ##  Homomorphic Evaluation

# %%

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import DensityMatrix, Statevector

def epr_eval_no_fhe(circuit, enc_psi):
    """
    Perform a circuit decomposed into Clifford + T gates on the encrypted quantum state
    homomorphically, using Qiskit.

    Args: 
        - circuit: QuantumCircuit containing the decomposed Clifford + T gates.
        - enc_psi: Statevector representing the encrypted pure state |psi>_{enc}.

    Returns: 
        - rho_hom_eval: DensityMatrix of the encrypted evaluated quantum state.
        - c_list: List of bits resulting from T-gate measurements.
    """
    # Convert enc_psi to a DensityMatrix for evaluation
    rho_hom_eval = DensityMatrix(enc_psi)
    
    # Initialize lists and dictionaries for bookkeeping
    num_qubits = circuit.num_qubits
    c_list = [[] for _ in range(num_qubits)]
    
    # Prepare the EPR pairs (if needed) by adding qubits to the circuit
    t_gate_count = number_of_tgates(circuit)
    if t_gate_count > 0:
        ancilla = QuantumRegister(2 * t_gate_count, "ancilla")
        circuit.add_register(ancilla)

    # Define gate functions mapping for common Clifford gates
    gate_functions = {
        "x": circuit.x,
        "z": circuit.z,
        "h": circuit.h,
        "s": circuit.s,
        # Add more gates if needed
    }

    for i, instruction in enumerate(circuit.data[:10]):  # Process only the first 10 gates
        gate = instruction.operation
        qubits = instruction.qubits
        gate_func = gate_functions.get(gate.name)

        # Retrieve the actual index of the qubit in the circuit
        qubit_index = circuit.qubits.index(qubits[0])

        print(f"Applying {gate.name} on qubit {qubit_index}")  # Log each gate application

        if gate_func:
            gate_func(qubit_index)  # Apply the gate function
        elif gate.name == "cx":  # Handle CNOT separately
            control_index = circuit.qubits.index(qubits[0])
            target_index = circuit.qubits.index(qubits[1])
            circuit.cx(control_index, target_index)
        elif gate.name == "t":  # Special handling for T-gates
            circuit.t(qubit_index)
            ancilla_qubit = ancilla[2 * qubit_index]
            circuit.cx(qubit_index, ancilla_qubit)
            circuit.measure(ancilla_qubit, qubit_index)
            c_list[qubit_index].append(qubit_index)


    # Convert final state to DensityMatrix and return measurement results
    rho_hom_eval = DensityMatrix.from_instruction(circuit)
    return rho_hom_eval, c_list

def epr_dec_no_fhe(rho_hom_eval, circuit, a, b, c_list):
    """
    Perform decryption of the encrypted evaluated cipherstate according to the EPR scheme without classical homomorphic encryption.

    Args: 
        - rho_hom_eval: DensityMatrix of the encrypted evaluated quantum state.
        - circuit: QuantumCircuit containing the Clifford + T gate decomposition.
        - a: List of bits for X-Pauli gates in the QOTP encryption.
        - b: List of bits for Z-Pauli gates in the QOTP encryption.
        - c_list: List of bits for updating the keys based on measurements.
        
    Returns: 
        - rho_dec_eval: Decrypted DensityMatrix of the input message.
    """
    # Initialize state and key tracking variables
    num_qubits = circuit.num_qubits
    rho_dec_eval = rho_hom_eval
    k_list = [[] for _ in range(num_qubits)]
    a_copy, b_copy = a.copy(), b.copy()

    # Track ancillary qubits if there are T-gates
    t_gate_count = number_of_tgates(circuit)
    if t_gate_count > 0:
        ancilla = QuantumRegister(2 * t_gate_count, "ancilla")
        circuit.add_register(ancilla)

    # Iterate through the circuit gates and apply the required decryption steps
    for gate, qubits, _ in circuit.data:
        qubit_index = circuit.qubits.index(qubits[0])

        if gate.name in {"i", "x", "z"}:
            pass  # No key updates required for these gates
        
        elif gate.name == "p":  # Phase gate key update
            b_copy[qubit_index] ^= a_copy[qubit_index]
        
        elif gate.name == "h":  # Hadamard gate key swap
            a_copy[qubit_index], b_copy[qubit_index] = b_copy[qubit_index], a_copy[qubit_index]
        
        elif gate.name == "cx":  # CNOT gate key update
            target_qubit_index = circuit.qubits.index(qubits[1])
            b_copy[qubit_index] ^= b_copy[target_qubit_index]
            a_copy[target_qubit_index] ^= a_copy[qubit_index]
        
        elif gate.name == "t":  # T gate key update and measurement
            # Apply T gate on the main qubit and controlled operation on the ancilla qubit
            ancilla_qubit = ancilla[qubit_index]
            circuit.t(qubit_index)
            circuit.cx(qubit_index, ancilla_qubit)
            circuit.h(ancilla_qubit)  # Hadamard for measurement basis change
            circuit.measure(ancilla_qubit, ClassicalRegister(1))

            # Obtain measurement result for the decryption key update
            measurement_result = int(c_list[qubit_index][0]) if c_list[qubit_index] else 0
            k_list[qubit_index].append(measurement_result)
            
            # Key update for T-gate with measurement outcome k
            k = measurement_result
            b_copy[qubit_index] ^= (a_copy[qubit_index] * c_list[qubit_index][0]) ^ k
            a_copy[qubit_index] ^= c_list[qubit_index][0]

    # After applying key updates, perform Quantum One-Time Pad (QOTP) for final decryption
    rho_dec_eval = qotp_dec(rho_dec_eval, circuit.qubits, a_copy, b_copy)

    return rho_dec_eval



# %% [markdown]
# ## Execution flow :
# 

# %%
def epr_qhe_no_fhe(psi, a, b, circuit):
    """
    Homomorphically apply the QuantumCircuit `circuit` to the encrypted quantum pure state |psi>
    using the EPR scheme, without classical fully homomorphic encryption.

    Args:
        psi: Statevector representing the pure state |psi>.
        a: List of (randomly generated) classical bits for QOTP X-Pauli encryption.
        b: List of (randomly generated) classical bits for QOTP Z-Pauli encryption.
        circuit: QuantumCircuit object containing the quantum operations to apply.

    Returns:
        rho_dec_eval: Decrypted DensityMatrix of the evaluated quantum state.
    """
    # Step 1: Apply Quantum One-Time Pad (QOTP) encryption
    psi_enc = qotp_enc(psi, psi.qubits, a, b)
    
    # Step 2: Homomorphically evaluate the encrypted circuit
    rho_hom_eval, c_list = epr_eval_no_fhe(circuit, psi_enc)
    
    # Step 3: Decrypt the evaluated circuit output
    rho_dec_eval = epr_dec_no_fhe(rho_hom_eval, circuit, a, b, c_list)
    
    return rho_dec_eval

# %%
psi = zero_state()  # Assuming zero_state() returns a QuantumCircuit
input_state_dimension = psi.num_qubits
a = [random.getrandbits(1) for _ in range(input_state_dimension)]
b = [random.getrandbits(1) for _ in range(input_state_dimension)]

# Define the circuit as a QuantumCircuit, not a list
circuit = QuantumCircuit(input_state_dimension)
circuit.x(0)  # Apply an X gate, or modify as needed for your evaluation
print(circuit)  # Print the circuit to verify its structure
print(epr_qhe_no_fhe(psi, a, b, circuit))

 


# %% [markdown]
# ### Prevent Repeated Application of the Same Gate on the Same Qubit To avoid applying the same gate to the same qubit multiple times, add a tracking mechanism using a set. This will ensure each gate is applied to a qubit only once.
# ### Replace your original for loop in epr_eval_no_fhe with this version, which includes a check using applied_gates:
# 

# %%

applied_gates = set()  # Set to track applied gate-qubit combinations

for instruction in circuit.data:
    gate = instruction.operation
    qubits = instruction.qubits
    qubit_index = circuit.qubits.index(qubits[0])

    # Create a unique key for the gate-qubit combination
    gate_key = (gate.name, qubit_index)
    if gate_key in applied_gates:
        continue  # Skip if already applied
    applied_gates.add(gate_key)

    print(f"Applying {gate.name} on qubit {qubit_index}")  # Log each gate application

    # Apply the gate
    gate_func = gate_functions.get(gate.name)
    if gate_func:
        gate_func(qubit_index)
    elif gate.name == "cx":  # Handle CNOT separately
        control_index = circuit.qubits.index(qubits[0])
        target_index = circuit.qubits.index(qubits[1])
        circuit.cx(control_index, target_index)
    elif gate.name == "t":  # Special handling for T-gates
        circuit.t(qubit_index)
        ancilla_qubit = ancilla[2 * qubit_index]
        circuit.cx(qubit_index, ancilla_qubit)
        circuit.measure(ancilla_qubit, qubit_index)
        c_list[qubit_index].append(qubit_index)

# %%
import os

# change the directory to run the desired notebook 
relative_path = 'epr-no-fhe'
current_dir = os.getcwd()
epr_no_fhe_dir = os.path.join(current_dir, relative_path)
os.chdir(epr_no_fhe_dir)

# %%
psi = zero_state()
input_state_dimension = int(np.log2(psi.shape[0]))
a = [random.getrandbits(1) for i in range(input_state_dimension)]
b =  [random.getrandbits(1) for i in range(input_state_dimension)]
circuit = ["X"]
print(epr_qhe_no_fhe(psi, a, b, circuit))

# %%
alpha = 0.3
beta = np.sqrt(1-alpha**2)
psi = gen_qubit(alpha, beta)
print(psi)


