
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

	orientation(lb2gt_0,141,108). coords(lb2gt_0,194,123,108).  visible(lb2gt_0,108). happensAt(active(lb2gt_0),108). orientation(lb2gt_3,146,108). coords(lb2gt_3,10,53,108).  visible(lb2gt_3,108). happensAt(walking(lb2gt_3),108).
orientation(lb2gt_0,141,109). coords(lb2gt_0,194,123,109).  visible(lb2gt_0,109). happensAt(active(lb2gt_0),109). orientation(lb2gt_3,146,109). coords(lb2gt_3,10,53,109).  visible(lb2gt_3,109). happensAt(walking(lb2gt_3),109).
orientation(lb2gt_0,141,110). coords(lb2gt_0,194,123,110).  visible(lb2gt_0,110). happensAt(active(lb2gt_0),110). orientation(lb2gt_3,146,110). coords(lb2gt_3,10,54,110).  visible(lb2gt_3,110). happensAt(walking(lb2gt_3),110).
orientation(lb2gt_0,141,111). coords(lb2gt_0,194,123,111).  visible(lb2gt_0,111). happensAt(active(lb2gt_0),111). orientation(lb2gt_3,146,111). coords(lb2gt_3,10,54,111).  visible(lb2gt_3,111). happensAt(walking(lb2gt_3),111).
orientation(lb2gt_0,141,112). coords(lb2gt_0,194,123,112).  visible(lb2gt_0,112). happensAt(active(lb2gt_0),112). orientation(lb2gt_3,146,112). coords(lb2gt_3,10,54,112).  visible(lb2gt_3,112). happensAt(walking(lb2gt_3),112).
orientation(lb2gt_0,141,113). coords(lb2gt_0,194,123,113).  visible(lb2gt_0,113). happensAt(active(lb2gt_0),113). orientation(lb2gt_3,146,113). coords(lb2gt_3,10,54,113).  visible(lb2gt_3,113). happensAt(walking(lb2gt_3),113).
orientation(lb2gt_0,141,114). coords(lb2gt_0,194,123,114).  visible(lb2gt_0,114). happensAt(active(lb2gt_0),114). orientation(lb2gt_3,146,114). coords(lb2gt_3,10,54,114).  visible(lb2gt_3,114). happensAt(walking(lb2gt_3),114).
orientation(lb2gt_0,141,115). coords(lb2gt_0,194,123,115).  visible(lb2gt_0,115). happensAt(active(lb2gt_0),115). orientation(lb2gt_3,146,115). coords(lb2gt_3,10,54,115).  visible(lb2gt_3,115). happensAt(walking(lb2gt_3),115).
orientation(lb2gt_0,141,116). coords(lb2gt_0,194,123,116).  visible(lb2gt_0,116). happensAt(active(lb2gt_0),116). orientation(lb2gt_3,146,116). coords(lb2gt_3,9,55,116).  visible(lb2gt_3,116). happensAt(walking(lb2gt_3),116).
orientation(lb2gt_0,141,117). coords(lb2gt_0,194,123,117).  visible(lb2gt_0,117). happensAt(active(lb2gt_0),117). orientation(lb2gt_3,146,117). coords(lb2gt_3,9,55,117).  visible(lb2gt_3,117). happensAt(walking(lb2gt_3),117).
orientation(lb2gt_0,141,118). coords(lb2gt_0,194,123,118).  visible(lb2gt_0,118). happensAt(active(lb2gt_0),118). orientation(lb2gt_3,146,118). coords(lb2gt_3,9,56,118).  visible(lb2gt_3,118). happensAt(walking(lb2gt_3),118).
orientation(lb2gt_0,141,119). coords(lb2gt_0,194,123,119).  visible(lb2gt_0,119). happensAt(active(lb2gt_0),119). orientation(lb2gt_3,146,119). coords(lb2gt_3,9,56,119).  visible(lb2gt_3,119). happensAt(walking(lb2gt_3),119).
orientation(lb2gt_0,141,120). coords(lb2gt_0,194,123,120).  visible(lb2gt_0,120). happensAt(active(lb2gt_0),120). orientation(lb2gt_3,146,120). coords(lb2gt_3,9,56,120).  visible(lb2gt_3,120). happensAt(walking(lb2gt_3),120).
orientation(lb2gt_0,141,121). coords(lb2gt_0,194,123,121).  visible(lb2gt_0,121). happensAt(active(lb2gt_0),121). orientation(lb2gt_3,146,121). coords(lb2gt_3,8,56,121).  visible(lb2gt_3,121). happensAt(walking(lb2gt_3),121).
orientation(lb2gt_0,141,122). coords(lb2gt_0,194,123,122).  visible(lb2gt_0,122). happensAt(active(lb2gt_0),122). orientation(lb2gt_3,146,122). coords(lb2gt_3,8,56,122).  visible(lb2gt_3,122). happensAt(walking(lb2gt_3),122).
orientation(lb2gt_0,141,123). coords(lb2gt_0,194,123,123).  visible(lb2gt_0,123). happensAt(active(lb2gt_0),123). orientation(lb2gt_3,146,123). coords(lb2gt_3,7,57,123).  visible(lb2gt_3,123). happensAt(walking(lb2gt_3),123).
orientation(lb2gt_0,141,124). coords(lb2gt_0,194,123,124).  visible(lb2gt_0,124). happensAt(active(lb2gt_0),124). orientation(lb2gt_3,146,124). coords(lb2gt_3,7,57,124).  visible(lb2gt_3,124). happensAt(walking(lb2gt_3),124).
orientation(lb2gt_0,141,125). coords(lb2gt_0,194,123,125).  visible(lb2gt_0,125). happensAt(active(lb2gt_0),125). orientation(lb2gt_3,146,125). coords(lb2gt_3,7,57,125).  visible(lb2gt_3,125). happensAt(walking(lb2gt_3),125).
orientation(lb2gt_0,141,126). coords(lb2gt_0,194,123,126).  visible(lb2gt_0,126). happensAt(active(lb2gt_0),126). orientation(lb2gt_3,146,126). coords(lb2gt_3,7,57,126).  visible(lb2gt_3,126). happensAt(walking(lb2gt_3),126).
orientation(lb2gt_0,141,127). coords(lb2gt_0,194,123,127).  visible(lb2gt_0,127). happensAt(active(lb2gt_0),127). orientation(lb2gt_3,146,127). coords(lb2gt_3,7,58,127).  visible(lb2gt_3,127). happensAt(walking(lb2gt_3),127).
orientation(lb2gt_0,141,128). coords(lb2gt_0,194,123,128).  visible(lb2gt_0,128). happensAt(active(lb2gt_0),128). orientation(lb2gt_3,146,128). coords(lb2gt_3,7,58,128).  visible(lb2gt_3,128). happensAt(walking(lb2gt_3),128).
orientation(lb2gt_0,141,129). coords(lb2gt_0,194,125,129).  visible(lb2gt_0,129). happensAt(active(lb2gt_0),129). orientation(lb2gt_3,146,129). coords(lb2gt_3,7,58,129).  visible(lb2gt_3,129). happensAt(walking(lb2gt_3),129).
orientation(lb2gt_0,141,130). coords(lb2gt_0,194,125,130).  visible(lb2gt_0,130). happensAt(active(lb2gt_0),130). orientation(lb2gt_3,146,130). coords(lb2gt_3,7,59,130).  visible(lb2gt_3,130). happensAt(walking(lb2gt_3),130).
orientation(lb2gt_0,141,131). coords(lb2gt_0,194,126,131).  visible(lb2gt_0,131). happensAt(active(lb2gt_0),131). orientation(lb2gt_3,153,131). coords(lb2gt_3,7,59,131).  visible(lb2gt_3,131). happensAt(walking(lb2gt_3),131).
orientation(lb2gt_0,141,132). coords(lb2gt_0,193,126,132).  visible(lb2gt_0,132). happensAt(active(lb2gt_0),132). orientation(lb2gt_3,156,132). coords(lb2gt_3,7,59,132).  visible(lb2gt_3,132). happensAt(walking(lb2gt_3),132).
#show holdsAt/2.