
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lbgt_0,154,108). coords(lbgt_0,153,108,108).  visible(lbgt_0,108). happensAt(inactive(lbgt_0),108). orientation(lbgt_1,145,108). coords(lbgt_1,76,79,108).  visible(lbgt_1,108). happensAt(walking(lbgt_1),108).
orientation(lbgt_0,154,109). coords(lbgt_0,153,107,109).  visible(lbgt_0,109). happensAt(inactive(lbgt_0),109). orientation(lbgt_1,145,109). coords(lbgt_1,76,79,109).  visible(lbgt_1,109). happensAt(walking(lbgt_1),109).
orientation(lbgt_0,154,110). coords(lbgt_0,154,107,110).  visible(lbgt_0,110). happensAt(active(lbgt_0),110). orientation(lbgt_1,145,110). coords(lbgt_1,76,79,110).  visible(lbgt_1,110). happensAt(walking(lbgt_1),110).
orientation(lbgt_0,154,111). coords(lbgt_0,156,107,111).  visible(lbgt_0,111). happensAt(active(lbgt_0),111). orientation(lbgt_1,145,111). coords(lbgt_1,76,79,111).  visible(lbgt_1,111). happensAt(walking(lbgt_1),111).
orientation(lbgt_0,154,112). coords(lbgt_0,158,108,112).  visible(lbgt_0,112). happensAt(active(lbgt_0),112). orientation(lbgt_1,145,112). coords(lbgt_1,76,79,112).  visible(lbgt_1,112). happensAt(walking(lbgt_1),112).
orientation(lbgt_0,154,113). coords(lbgt_0,159,109,113).  visible(lbgt_0,113). happensAt(active(lbgt_0),113). orientation(lbgt_1,145,113). coords(lbgt_1,76,79,113).  visible(lbgt_1,113). happensAt(walking(lbgt_1),113).
orientation(lbgt_0,154,114). coords(lbgt_0,160,111,114).  visible(lbgt_0,114). happensAt(active(lbgt_0),114). orientation(lbgt_1,153,114). coords(lbgt_1,76,80,114).  visible(lbgt_1,114). happensAt(walking(lbgt_1),114).
orientation(lbgt_0,154,115). coords(lbgt_0,161,113,115).  visible(lbgt_0,115). happensAt(active(lbgt_0),115). orientation(lbgt_1,155,115). coords(lbgt_1,76,79,115).  visible(lbgt_1,115). happensAt(walking(lbgt_1),115).
orientation(lbgt_0,154,116). coords(lbgt_0,161,114,116).  visible(lbgt_0,116). happensAt(active(lbgt_0),116). orientation(lbgt_1,154,116). coords(lbgt_1,76,80,116).  visible(lbgt_1,116). happensAt(walking(lbgt_1),116).
orientation(lbgt_0,154,117). coords(lbgt_0,162,116,117).  visible(lbgt_0,117). happensAt(active(lbgt_0),117). orientation(lbgt_1,153,117). coords(lbgt_1,76,80,117).  visible(lbgt_1,117). happensAt(walking(lbgt_1),117).
orientation(lbgt_0,154,118). coords(lbgt_0,162,118,118).  visible(lbgt_0,118). happensAt(active(lbgt_0),118). orientation(lbgt_1,154,118). coords(lbgt_1,76,80,118).  visible(lbgt_1,118). happensAt(walking(lbgt_1),118).
orientation(lbgt_0,149,119). coords(lbgt_0,163,120,119).  visible(lbgt_0,119). happensAt(active(lbgt_0),119). orientation(lbgt_1,155,119). coords(lbgt_1,76,80,119).  visible(lbgt_1,119). happensAt(walking(lbgt_1),119).
orientation(lbgt_0,145,120). coords(lbgt_0,163,122,120).  visible(lbgt_0,120). happensAt(active(lbgt_0),120). orientation(lbgt_1,155,120). coords(lbgt_1,75,83,120).  visible(lbgt_1,120). happensAt(walking(lbgt_1),120).
orientation(lbgt_0,145,121). coords(lbgt_0,164,124,121).  visible(lbgt_0,121). happensAt(active(lbgt_0),121). orientation(lbgt_1,155,121). coords(lbgt_1,76,82,121).  visible(lbgt_1,121). happensAt(walking(lbgt_1),121).
orientation(lbgt_0,145,122). coords(lbgt_0,166,125,122).  visible(lbgt_0,122). happensAt(active(lbgt_0),122). orientation(lbgt_1,155,122). coords(lbgt_1,76,83,122).  visible(lbgt_1,122). happensAt(walking(lbgt_1),122).
orientation(lbgt_0,145,123). coords(lbgt_0,167,127,123).  visible(lbgt_0,123). happensAt(active(lbgt_0),123). orientation(lbgt_1,155,123). coords(lbgt_1,76,84,123).  visible(lbgt_1,123). happensAt(walking(lbgt_1),123).
orientation(lbgt_0,145,124). coords(lbgt_0,168,128,124).  visible(lbgt_0,124). happensAt(active(lbgt_0),124). orientation(lbgt_1,155,124). coords(lbgt_1,76,84,124).  visible(lbgt_1,124). happensAt(walking(lbgt_1),124).
orientation(lbgt_0,145,125). coords(lbgt_0,169,129,125).  visible(lbgt_0,125). happensAt(active(lbgt_0),125). orientation(lbgt_1,155,125). coords(lbgt_1,76,84,125).  visible(lbgt_1,125). happensAt(walking(lbgt_1),125).
orientation(lbgt_0,145,126). coords(lbgt_0,170,131,126).  visible(lbgt_0,126). happensAt(active(lbgt_0),126). orientation(lbgt_1,155,126). coords(lbgt_1,76,84,126).  visible(lbgt_1,126). happensAt(walking(lbgt_1),126).
orientation(lbgt_0,145,127). coords(lbgt_0,170,132,127).  visible(lbgt_0,127). happensAt(active(lbgt_0),127). orientation(lbgt_1,155,127). coords(lbgt_1,76,84,127).  visible(lbgt_1,127). happensAt(walking(lbgt_1),127).
orientation(lbgt_0,151,128). coords(lbgt_0,171,133,128).  visible(lbgt_0,128). happensAt(walking(lbgt_0),128). orientation(lbgt_1,155,128). coords(lbgt_1,76,84,128).  visible(lbgt_1,128). happensAt(walking(lbgt_1),128).
orientation(lbgt_0,155,129). coords(lbgt_0,171,134,129).  visible(lbgt_0,129). happensAt(walking(lbgt_0),129). orientation(lbgt_1,155,129). coords(lbgt_1,76,84,129).  visible(lbgt_1,129). happensAt(walking(lbgt_1),129).
orientation(lbgt_0,157,130). coords(lbgt_0,171,135,130).  visible(lbgt_0,130). happensAt(walking(lbgt_0),130). orientation(lbgt_1,155,130). coords(lbgt_1,76,85,130).  visible(lbgt_1,130). happensAt(walking(lbgt_1),130).
orientation(lbgt_0,159,131). coords(lbgt_0,172,136,131).  visible(lbgt_0,131). happensAt(walking(lbgt_0),131). orientation(lbgt_1,155,131). coords(lbgt_1,76,85,131).  visible(lbgt_1,131). happensAt(walking(lbgt_1),131).
orientation(lbgt_0,159,132). coords(lbgt_0,174,137,132).  visible(lbgt_0,132). happensAt(walking(lbgt_0),132). orientation(lbgt_1,155,132). coords(lbgt_1,76,86,132).  visible(lbgt_1,132). happensAt(walking(lbgt_1),132).
#show holdsAt/2.