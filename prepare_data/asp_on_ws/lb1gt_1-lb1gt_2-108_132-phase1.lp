
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

	orientation(lb1gt_1,151,108). coords(lb1gt_1,79,120,108).  visible(lb1gt_1,108). happensAt(walking(lb1gt_1),108). orientation(lb1gt_2,147,108). coords(lb1gt_2,93,90,108).  visible(lb1gt_2,108). happensAt(walking(lb1gt_2),108).
orientation(lb1gt_1,151,109). coords(lb1gt_1,80,121,109).  visible(lb1gt_1,109). happensAt(walking(lb1gt_1),109). orientation(lb1gt_2,147,109). coords(lb1gt_2,93,90,109).  visible(lb1gt_2,109). happensAt(walking(lb1gt_2),109).
orientation(lb1gt_1,151,110). coords(lb1gt_1,81,121,110).  visible(lb1gt_1,110). happensAt(walking(lb1gt_1),110). orientation(lb1gt_2,147,110). coords(lb1gt_2,93,90,110).  visible(lb1gt_2,110). happensAt(walking(lb1gt_2),110).
orientation(lb1gt_1,151,111). coords(lb1gt_1,81,122,111).  visible(lb1gt_1,111). happensAt(walking(lb1gt_1),111). orientation(lb1gt_2,147,111). coords(lb1gt_2,93,90,111).  visible(lb1gt_2,111). happensAt(walking(lb1gt_2),111).
orientation(lb1gt_1,151,112). coords(lb1gt_1,81,123,112).  visible(lb1gt_1,112). happensAt(walking(lb1gt_1),112). orientation(lb1gt_2,147,112). coords(lb1gt_2,94,90,112).  visible(lb1gt_2,112). happensAt(walking(lb1gt_2),112).
orientation(lb1gt_1,151,113). coords(lb1gt_1,82,123,113).  visible(lb1gt_1,113). happensAt(walking(lb1gt_1),113). orientation(lb1gt_2,147,113). coords(lb1gt_2,94,91,113).  visible(lb1gt_2,113). happensAt(walking(lb1gt_2),113).
orientation(lb1gt_1,151,114). coords(lb1gt_1,83,123,114).  visible(lb1gt_1,114). happensAt(walking(lb1gt_1),114). orientation(lb1gt_2,147,114). coords(lb1gt_2,94,92,114).  visible(lb1gt_2,114). happensAt(walking(lb1gt_2),114).
orientation(lb1gt_1,151,115). coords(lb1gt_1,83,123,115).  visible(lb1gt_1,115). happensAt(walking(lb1gt_1),115). orientation(lb1gt_2,147,115). coords(lb1gt_2,94,92,115).  visible(lb1gt_2,115). happensAt(walking(lb1gt_2),115).
orientation(lb1gt_1,151,116). coords(lb1gt_1,83,124,116).  visible(lb1gt_1,116). happensAt(walking(lb1gt_1),116). orientation(lb1gt_2,147,116). coords(lb1gt_2,94,94,116).  visible(lb1gt_2,116). happensAt(walking(lb1gt_2),116).
orientation(lb1gt_1,151,117). coords(lb1gt_1,83,124,117).  visible(lb1gt_1,117). happensAt(walking(lb1gt_1),117). orientation(lb1gt_2,147,117). coords(lb1gt_2,94,94,117).  visible(lb1gt_2,117). happensAt(walking(lb1gt_2),117).
orientation(lb1gt_1,151,118). coords(lb1gt_1,83,124,118).  visible(lb1gt_1,118). happensAt(walking(lb1gt_1),118). orientation(lb1gt_2,147,118). coords(lb1gt_2,94,95,118).  visible(lb1gt_2,118). happensAt(walking(lb1gt_2),118).
orientation(lb1gt_1,151,119). coords(lb1gt_1,83,124,119).  visible(lb1gt_1,119). happensAt(walking(lb1gt_1),119). orientation(lb1gt_2,147,119). coords(lb1gt_2,94,96,119).  visible(lb1gt_2,119). happensAt(walking(lb1gt_2),119).
orientation(lb1gt_1,151,120). coords(lb1gt_1,84,124,120).  visible(lb1gt_1,120). happensAt(walking(lb1gt_1),120). orientation(lb1gt_2,147,120). coords(lb1gt_2,94,96,120).  visible(lb1gt_2,120). happensAt(walking(lb1gt_2),120).
orientation(lb1gt_1,151,121). coords(lb1gt_1,84,124,121).  visible(lb1gt_1,121). happensAt(walking(lb1gt_1),121). orientation(lb1gt_2,147,121). coords(lb1gt_2,94,96,121).  visible(lb1gt_2,121). happensAt(walking(lb1gt_2),121).
orientation(lb1gt_1,151,122). coords(lb1gt_1,84,125,122).  visible(lb1gt_1,122). happensAt(walking(lb1gt_1),122). orientation(lb1gt_2,147,122). coords(lb1gt_2,94,97,122).  visible(lb1gt_2,122). happensAt(walking(lb1gt_2),122).
orientation(lb1gt_1,151,123). coords(lb1gt_1,84,126,123).  visible(lb1gt_1,123). happensAt(walking(lb1gt_1),123). orientation(lb1gt_2,147,123). coords(lb1gt_2,94,97,123).  visible(lb1gt_2,123). happensAt(walking(lb1gt_2),123).
orientation(lb1gt_1,151,124). coords(lb1gt_1,84,127,124).  visible(lb1gt_1,124). happensAt(walking(lb1gt_1),124). orientation(lb1gt_2,147,124). coords(lb1gt_2,94,98,124).  visible(lb1gt_2,124). happensAt(walking(lb1gt_2),124).
orientation(lb1gt_1,151,125). coords(lb1gt_1,84,129,125).  visible(lb1gt_1,125). happensAt(walking(lb1gt_1),125). orientation(lb1gt_2,147,125). coords(lb1gt_2,94,98,125).  visible(lb1gt_2,125). happensAt(walking(lb1gt_2),125).
orientation(lb1gt_1,151,126). coords(lb1gt_1,84,130,126).  visible(lb1gt_1,126). happensAt(walking(lb1gt_1),126). orientation(lb1gt_2,147,126). coords(lb1gt_2,94,98,126).  visible(lb1gt_2,126). happensAt(walking(lb1gt_2),126).
orientation(lb1gt_1,151,127). coords(lb1gt_1,85,131,127).  visible(lb1gt_1,127). happensAt(walking(lb1gt_1),127). orientation(lb1gt_2,144,127). coords(lb1gt_2,95,98,127).  visible(lb1gt_2,127). happensAt(walking(lb1gt_2),127).
orientation(lb1gt_1,151,128). coords(lb1gt_1,86,132,128).  visible(lb1gt_1,128). happensAt(walking(lb1gt_1),128). orientation(lb1gt_2,144,128). coords(lb1gt_2,95,98,128).  visible(lb1gt_2,128). happensAt(walking(lb1gt_2),128).
orientation(lb1gt_1,151,129). coords(lb1gt_1,86,132,129).  visible(lb1gt_1,129). happensAt(walking(lb1gt_1),129). orientation(lb1gt_2,144,129). coords(lb1gt_2,95,98,129).  visible(lb1gt_2,129). happensAt(walking(lb1gt_2),129).
orientation(lb1gt_1,151,130). coords(lb1gt_1,86,133,130).  visible(lb1gt_1,130). happensAt(walking(lb1gt_1),130). orientation(lb1gt_2,144,130). coords(lb1gt_2,95,99,130).  visible(lb1gt_2,130). happensAt(walking(lb1gt_2),130).
orientation(lb1gt_1,151,131). coords(lb1gt_1,86,133,131).  visible(lb1gt_1,131). happensAt(walking(lb1gt_1),131). orientation(lb1gt_2,144,131). coords(lb1gt_2,96,99,131).  visible(lb1gt_2,131). happensAt(walking(lb1gt_2),131).
orientation(lb1gt_1,151,132). coords(lb1gt_1,87,133,132).  visible(lb1gt_1,132). happensAt(walking(lb1gt_1),132). orientation(lb1gt_2,144,132). coords(lb1gt_2,96,99,132).  visible(lb1gt_2,132). happensAt(walking(lb1gt_2),132).
#show holdsAt/2.