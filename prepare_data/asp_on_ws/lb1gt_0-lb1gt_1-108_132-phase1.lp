
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

	orientation(lb1gt_0,147,108). coords(lb1gt_0,177,115,108).  visible(lb1gt_0,108). happensAt(active(lb1gt_0),108). orientation(lb1gt_1,151,108). coords(lb1gt_1,79,120,108).  visible(lb1gt_1,108). happensAt(walking(lb1gt_1),108).
orientation(lb1gt_0,150,109). coords(lb1gt_0,178,116,109).  visible(lb1gt_0,109). happensAt(active(lb1gt_0),109). orientation(lb1gt_1,151,109). coords(lb1gt_1,80,121,109).  visible(lb1gt_1,109). happensAt(walking(lb1gt_1),109).
orientation(lb1gt_0,150,110). coords(lb1gt_0,179,117,110).  visible(lb1gt_0,110). happensAt(active(lb1gt_0),110). orientation(lb1gt_1,151,110). coords(lb1gt_1,81,121,110).  visible(lb1gt_1,110). happensAt(walking(lb1gt_1),110).
orientation(lb1gt_0,149,111). coords(lb1gt_0,179,118,111).  visible(lb1gt_0,111). happensAt(active(lb1gt_0),111). orientation(lb1gt_1,151,111). coords(lb1gt_1,81,122,111).  visible(lb1gt_1,111). happensAt(walking(lb1gt_1),111).
orientation(lb1gt_0,150,112). coords(lb1gt_0,180,118,112).  visible(lb1gt_0,112). happensAt(active(lb1gt_0),112). orientation(lb1gt_1,151,112). coords(lb1gt_1,81,123,112).  visible(lb1gt_1,112). happensAt(walking(lb1gt_1),112).
orientation(lb1gt_0,151,113). coords(lb1gt_0,180,119,113).  visible(lb1gt_0,113). happensAt(active(lb1gt_0),113). orientation(lb1gt_1,151,113). coords(lb1gt_1,82,123,113).  visible(lb1gt_1,113). happensAt(walking(lb1gt_1),113).
orientation(lb1gt_0,151,114). coords(lb1gt_0,180,119,114).  visible(lb1gt_0,114). happensAt(active(lb1gt_0),114). orientation(lb1gt_1,151,114). coords(lb1gt_1,83,123,114).  visible(lb1gt_1,114). happensAt(walking(lb1gt_1),114).
orientation(lb1gt_0,151,115). coords(lb1gt_0,180,119,115).  visible(lb1gt_0,115). happensAt(active(lb1gt_0),115). orientation(lb1gt_1,151,115). coords(lb1gt_1,83,123,115).  visible(lb1gt_1,115). happensAt(walking(lb1gt_1),115).
orientation(lb1gt_0,151,116). coords(lb1gt_0,180,120,116).  visible(lb1gt_0,116). happensAt(active(lb1gt_0),116). orientation(lb1gt_1,151,116). coords(lb1gt_1,83,124,116).  visible(lb1gt_1,116). happensAt(walking(lb1gt_1),116).
orientation(lb1gt_0,151,117). coords(lb1gt_0,180,120,117).  visible(lb1gt_0,117). happensAt(active(lb1gt_0),117). orientation(lb1gt_1,151,117). coords(lb1gt_1,83,124,117).  visible(lb1gt_1,117). happensAt(walking(lb1gt_1),117).
orientation(lb1gt_0,151,118). coords(lb1gt_0,180,120,118).  visible(lb1gt_0,118). happensAt(active(lb1gt_0),118). orientation(lb1gt_1,151,118). coords(lb1gt_1,83,124,118).  visible(lb1gt_1,118). happensAt(walking(lb1gt_1),118).
orientation(lb1gt_0,151,119). coords(lb1gt_0,180,120,119).  visible(lb1gt_0,119). happensAt(active(lb1gt_0),119). orientation(lb1gt_1,151,119). coords(lb1gt_1,83,124,119).  visible(lb1gt_1,119). happensAt(walking(lb1gt_1),119).
orientation(lb1gt_0,151,120). coords(lb1gt_0,180,120,120).  visible(lb1gt_0,120). happensAt(active(lb1gt_0),120). orientation(lb1gt_1,151,120). coords(lb1gt_1,84,124,120).  visible(lb1gt_1,120). happensAt(walking(lb1gt_1),120).
orientation(lb1gt_0,151,121). coords(lb1gt_0,180,120,121).  visible(lb1gt_0,121). happensAt(active(lb1gt_0),121). orientation(lb1gt_1,151,121). coords(lb1gt_1,84,124,121).  visible(lb1gt_1,121). happensAt(walking(lb1gt_1),121).
orientation(lb1gt_0,151,122). coords(lb1gt_0,180,120,122).  visible(lb1gt_0,122). happensAt(active(lb1gt_0),122). orientation(lb1gt_1,151,122). coords(lb1gt_1,84,125,122).  visible(lb1gt_1,122). happensAt(walking(lb1gt_1),122).
orientation(lb1gt_0,151,123). coords(lb1gt_0,180,119,123).  visible(lb1gt_0,123). happensAt(active(lb1gt_0),123). orientation(lb1gt_1,151,123). coords(lb1gt_1,84,126,123).  visible(lb1gt_1,123). happensAt(walking(lb1gt_1),123).
orientation(lb1gt_0,149,124). coords(lb1gt_0,180,119,124).  visible(lb1gt_0,124). happensAt(active(lb1gt_0),124). orientation(lb1gt_1,151,124). coords(lb1gt_1,84,127,124).  visible(lb1gt_1,124). happensAt(walking(lb1gt_1),124).
orientation(lb1gt_0,146,125). coords(lb1gt_0,180,119,125).  visible(lb1gt_0,125). happensAt(active(lb1gt_0),125). orientation(lb1gt_1,151,125). coords(lb1gt_1,84,129,125).  visible(lb1gt_1,125). happensAt(walking(lb1gt_1),125).
orientation(lb1gt_0,145,126). coords(lb1gt_0,180,117,126).  visible(lb1gt_0,126). happensAt(walking(lb1gt_0),126). orientation(lb1gt_1,151,126). coords(lb1gt_1,84,130,126).  visible(lb1gt_1,126). happensAt(walking(lb1gt_1),126).
orientation(lb1gt_0,145,127). coords(lb1gt_0,179,116,127).  visible(lb1gt_0,127). happensAt(walking(lb1gt_0),127). orientation(lb1gt_1,151,127). coords(lb1gt_1,85,131,127).  visible(lb1gt_1,127). happensAt(walking(lb1gt_1),127).
orientation(lb1gt_0,144,128). coords(lb1gt_0,179,115,128).  visible(lb1gt_0,128). happensAt(walking(lb1gt_0),128). orientation(lb1gt_1,151,128). coords(lb1gt_1,86,132,128).  visible(lb1gt_1,128). happensAt(walking(lb1gt_1),128).
orientation(lb1gt_0,143,129). coords(lb1gt_0,179,114,129).  visible(lb1gt_0,129). happensAt(walking(lb1gt_0),129). orientation(lb1gt_1,151,129). coords(lb1gt_1,86,132,129).  visible(lb1gt_1,129). happensAt(walking(lb1gt_1),129).
orientation(lb1gt_0,142,130). coords(lb1gt_0,179,113,130).  visible(lb1gt_0,130). happensAt(walking(lb1gt_0),130). orientation(lb1gt_1,151,130). coords(lb1gt_1,86,133,130).  visible(lb1gt_1,130). happensAt(walking(lb1gt_1),130).
orientation(lb1gt_0,142,131). coords(lb1gt_0,178,112,131).  visible(lb1gt_0,131). happensAt(walking(lb1gt_0),131). orientation(lb1gt_1,151,131). coords(lb1gt_1,86,133,131).  visible(lb1gt_1,131). happensAt(walking(lb1gt_1),131).
orientation(lb1gt_0,142,132). coords(lb1gt_0,177,111,132).  visible(lb1gt_0,132). happensAt(walking(lb1gt_0),132). orientation(lb1gt_1,151,132). coords(lb1gt_1,87,133,132).  visible(lb1gt_1,132). happensAt(walking(lb1gt_1),132).
#show holdsAt/2.