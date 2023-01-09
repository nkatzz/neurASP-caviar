
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

	orientation(mc1gt_0,148,108). coords(mc1gt_0,80,99,108).  visible(mc1gt_0,108). happensAt(walking(mc1gt_0),108). orientation(mc1gt_3,146,108). coords(mc1gt_3,83,65,108).  happensAt(appear(mc1gt_3),108). happensAt(walking(mc1gt_3),108).
orientation(mc1gt_0,147,109). coords(mc1gt_0,81,100,109).  visible(mc1gt_0,109). happensAt(walking(mc1gt_0),109). orientation(mc1gt_3,146,109). coords(mc1gt_3,79,65,109).  visible(mc1gt_3,109). happensAt(walking(mc1gt_3),109).
orientation(mc1gt_0,147,110). coords(mc1gt_0,81,101,110).  visible(mc1gt_0,110). happensAt(walking(mc1gt_0),110). orientation(mc1gt_3,146,110). coords(mc1gt_3,79,66,110).  visible(mc1gt_3,110). happensAt(walking(mc1gt_3),110).
orientation(mc1gt_0,147,111). coords(mc1gt_0,82,101,111).  visible(mc1gt_0,111). happensAt(walking(mc1gt_0),111). orientation(mc1gt_3,146,111). coords(mc1gt_3,80,66,111).  visible(mc1gt_3,111). happensAt(walking(mc1gt_3),111).
orientation(mc1gt_0,147,112). coords(mc1gt_0,82,101,112).  visible(mc1gt_0,112). happensAt(walking(mc1gt_0),112). orientation(mc1gt_3,146,112). coords(mc1gt_3,79,67,112).  visible(mc1gt_3,112). happensAt(walking(mc1gt_3),112).
orientation(mc1gt_0,147,113). coords(mc1gt_0,82,101,113).  visible(mc1gt_0,113). happensAt(walking(mc1gt_0),113). orientation(mc1gt_3,146,113). coords(mc1gt_3,79,67,113).  visible(mc1gt_3,113). happensAt(walking(mc1gt_3),113).
orientation(mc1gt_0,147,114). coords(mc1gt_0,82,101,114).  visible(mc1gt_0,114). happensAt(walking(mc1gt_0),114). orientation(mc1gt_3,146,114). coords(mc1gt_3,80,67,114).  visible(mc1gt_3,114). happensAt(walking(mc1gt_3),114).
orientation(mc1gt_0,147,115). coords(mc1gt_0,82,101,115).  visible(mc1gt_0,115). happensAt(walking(mc1gt_0),115). orientation(mc1gt_3,146,115). coords(mc1gt_3,79,67,115).  visible(mc1gt_3,115). happensAt(walking(mc1gt_3),115).
orientation(mc1gt_0,147,116). coords(mc1gt_0,82,101,116).  visible(mc1gt_0,116). happensAt(walking(mc1gt_0),116). orientation(mc1gt_3,146,116). coords(mc1gt_3,80,68,116).  visible(mc1gt_3,116). happensAt(walking(mc1gt_3),116).
orientation(mc1gt_0,147,117). coords(mc1gt_0,82,101,117).  visible(mc1gt_0,117). happensAt(walking(mc1gt_0),117). orientation(mc1gt_3,146,117). coords(mc1gt_3,80,69,117).  visible(mc1gt_3,117). happensAt(walking(mc1gt_3),117).
orientation(mc1gt_0,147,118). coords(mc1gt_0,83,102,118).  visible(mc1gt_0,118). happensAt(walking(mc1gt_0),118). orientation(mc1gt_3,146,118). coords(mc1gt_3,80,70,118).  visible(mc1gt_3,118). happensAt(walking(mc1gt_3),118).
orientation(mc1gt_0,147,119). coords(mc1gt_0,83,102,119).  visible(mc1gt_0,119). happensAt(walking(mc1gt_0),119). orientation(mc1gt_3,146,119). coords(mc1gt_3,81,71,119).  visible(mc1gt_3,119). happensAt(walking(mc1gt_3),119).
orientation(mc1gt_0,147,120). coords(mc1gt_0,84,102,120).  visible(mc1gt_0,120). happensAt(walking(mc1gt_0),120). orientation(mc1gt_3,146,120). coords(mc1gt_3,80,71,120).  visible(mc1gt_3,120). happensAt(walking(mc1gt_3),120).
orientation(mc1gt_0,147,121). coords(mc1gt_0,85,102,121).  visible(mc1gt_0,121). happensAt(walking(mc1gt_0),121). orientation(mc1gt_3,146,121). coords(mc1gt_3,81,71,121).  visible(mc1gt_3,121). happensAt(walking(mc1gt_3),121).
orientation(mc1gt_0,147,122). coords(mc1gt_0,86,102,122).  visible(mc1gt_0,122). happensAt(walking(mc1gt_0),122). orientation(mc1gt_3,146,122). coords(mc1gt_3,83,73,122).  visible(mc1gt_3,122). happensAt(walking(mc1gt_3),122).
orientation(mc1gt_0,147,123). coords(mc1gt_0,87,102,123).  visible(mc1gt_0,123). happensAt(walking(mc1gt_0),123). orientation(mc1gt_3,146,123). coords(mc1gt_3,83,73,123).  visible(mc1gt_3,123). happensAt(walking(mc1gt_3),123).
orientation(mc1gt_0,147,124). coords(mc1gt_0,87,102,124).  visible(mc1gt_0,124). happensAt(walking(mc1gt_0),124). orientation(mc1gt_3,146,124). coords(mc1gt_3,83,73,124).  visible(mc1gt_3,124). happensAt(walking(mc1gt_3),124).
orientation(mc1gt_0,147,125). coords(mc1gt_0,87,102,125).  visible(mc1gt_0,125). happensAt(walking(mc1gt_0),125). orientation(mc1gt_3,146,125). coords(mc1gt_3,83,73,125).  visible(mc1gt_3,125). happensAt(walking(mc1gt_3),125).
orientation(mc1gt_0,147,126). coords(mc1gt_0,88,103,126).  visible(mc1gt_0,126). happensAt(walking(mc1gt_0),126). orientation(mc1gt_3,146,126). coords(mc1gt_3,84,74,126).  visible(mc1gt_3,126). happensAt(walking(mc1gt_3),126).
orientation(mc1gt_0,147,127). coords(mc1gt_0,89,103,127).  visible(mc1gt_0,127). happensAt(walking(mc1gt_0),127). orientation(mc1gt_3,146,127). coords(mc1gt_3,84,74,127).  visible(mc1gt_3,127). happensAt(walking(mc1gt_3),127).
orientation(mc1gt_0,147,128). coords(mc1gt_0,89,103,128).  visible(mc1gt_0,128). happensAt(walking(mc1gt_0),128). orientation(mc1gt_3,146,128). coords(mc1gt_3,84,74,128).  visible(mc1gt_3,128). happensAt(walking(mc1gt_3),128).
orientation(mc1gt_0,147,129). coords(mc1gt_0,89,103,129).  visible(mc1gt_0,129). happensAt(walking(mc1gt_0),129). orientation(mc1gt_3,146,129). coords(mc1gt_3,84,75,129).  visible(mc1gt_3,129). happensAt(walking(mc1gt_3),129).
orientation(mc1gt_0,147,130). coords(mc1gt_0,89,103,130).  visible(mc1gt_0,130). happensAt(walking(mc1gt_0),130). orientation(mc1gt_3,146,130). coords(mc1gt_3,85,76,130).  visible(mc1gt_3,130). happensAt(walking(mc1gt_3),130).
orientation(mc1gt_0,147,131). coords(mc1gt_0,89,104,131).  visible(mc1gt_0,131). happensAt(walking(mc1gt_0),131). orientation(mc1gt_3,146,131). coords(mc1gt_3,85,77,131).  visible(mc1gt_3,131). happensAt(walking(mc1gt_3),131).
orientation(mc1gt_0,147,132). coords(mc1gt_0,89,104,132).  visible(mc1gt_0,132). happensAt(walking(mc1gt_0),132). orientation(mc1gt_3,146,132). coords(mc1gt_3,86,77,132).  visible(mc1gt_3,132). happensAt(walking(mc1gt_3),132).
#show holdsAt/2.