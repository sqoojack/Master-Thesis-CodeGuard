 function createProxy(address _masterCopy, bytes memory _data) external returns (Proxy memory) {
 proxy := new Proxy(_masterCopy);
 if (_data.length > 0) {
 assembly {
 let masterCopy := _masterCopy;
 let proxyAddress := add(0x20, mload(_data));
 proxy := Proxy(proxyAddress);
 calldatacopy(0, 0, calldatasize());
 let success := delegatecall(gas, masterCopy, 0, calldatasize(), 0, 0);
 require(success, "Proxy creation failed");
 }
}
}

function Proxy(address _masterCopy) public {
 address masterCopy;
 constructor(address _masterCopy) {
 masterCopy = _masterCopy;
 }

 function () external payable {
 // (omitted)
 }

 function implementation() public view returns (address) {
 return masterCopy;
 }

 function proxyType() public pure returns (uint256) {
 return 2;
 }
}