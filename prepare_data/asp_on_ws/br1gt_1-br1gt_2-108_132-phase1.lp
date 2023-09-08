
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

	orientation(br1gt_1,164,108). coords(br1gt_1,69,165,108).  visible(br1gt_1,108). happensAt(walking(br1gt_1),108). orientation(br1gt_2,171,108). coords(br1gt_2,86,207,108).  visible(br1gt_2,108). happensAt(inactive(br1gt_2),108).
orientation(br1gt_1,164,109). coords(br1gt_1,69,165,109).  visible(br1gt_1,109). happensAt(walking(br1gt_1),109). orientation(br1gt_2,171,109). coords(br1gt_2,86,207,109).  visible(br1gt_2,109). happensAt(inactive(br1gt_2),109).
orientation(br1gt_1,164,110). coords(br1gt_1,67,164,110).  visible(br1gt_1,110). happensAt(walking(br1gt_1),110). orientation(br1gt_2,171,110). coords(br1gt_2,86,207,110).  visible(br1gt_2,110). happensAt(inactive(br1gt_2),110).
orientation(br1gt_1,164,111). coords(br1gt_1,66,164,111).  visible(br1gt_1,111). happensAt(walking(br1gt_1),111). orientation(br1gt_2,171,111). coords(br1gt_2,86,207,111).  visible(br1gt_2,111). happensAt(inactive(br1gt_2),111).
orientation(br1gt_1,164,112). coords(br1gt_1,66,164,112).  visible(br1gt_1,112). happensAt(walking(br1gt_1),112). orientation(br1gt_2,171,112). coords(br1gt_2,86,207,112).  visible(br1gt_2,112). happensAt(inactive(br1gt_2),112).
orientation(br1gt_1,164,113). coords(br1gt_1,66,164,113).  visible(br1gt_1,113). happensAt(walking(br1gt_1),113). orientation(br1gt_2,171,113). coords(br1gt_2,86,207,113).  visible(br1gt_2,113). happensAt(inactive(br1gt_2),113).
orientation(br1gt_1,164,114). coords(br1gt_1,66,164,114).  visible(br1gt_1,114). happensAt(walking(br1gt_1),114). orientation(br1gt_2,171,114). coords(br1gt_2,86,207,114).  visible(br1gt_2,114). happensAt(inactive(br1gt_2),114).
orientation(br1gt_1,164,115). coords(br1gt_1,66,164,115).  visible(br1gt_1,115). happensAt(walking(br1gt_1),115). orientation(br1gt_2,171,115). coords(br1gt_2,86,207,115).  visible(br1gt_2,115). happensAt(inactive(br1gt_2),115).
orientation(br1gt_1,164,116). coords(br1gt_1,66,164,116).  visible(br1gt_1,116). happensAt(walking(br1gt_1),116). orientation(br1gt_2,171,116). coords(br1gt_2,86,207,116).  visible(br1gt_2,116). happensAt(inactive(br1gt_2),116).
orientation(br1gt_1,164,117). coords(br1gt_1,66,164,117).  visible(br1gt_1,117). happensAt(walking(br1gt_1),117). orientation(br1gt_2,171,117). coords(br1gt_2,86,207,117).  visible(br1gt_2,117). happensAt(inactive(br1gt_2),117).
orientation(br1gt_1,164,118). coords(br1gt_1,65,164,118).  visible(br1gt_1,118). happensAt(walking(br1gt_1),118). orientation(br1gt_2,171,118). coords(br1gt_2,86,207,118).  visible(br1gt_2,118). happensAt(inactive(br1gt_2),118).
orientation(br1gt_1,164,119). coords(br1gt_1,63,164,119).  visible(br1gt_1,119). happensAt(walking(br1gt_1),119). orientation(br1gt_2,171,119). coords(br1gt_2,86,207,119).  visible(br1gt_2,119). happensAt(inactive(br1gt_2),119).
orientation(br1gt_1,164,120). coords(br1gt_1,62,163,120).  visible(br1gt_1,120). happensAt(walking(br1gt_1),120). orientation(br1gt_2,171,120). coords(br1gt_2,86,207,120).  visible(br1gt_2,120). happensAt(inactive(br1gt_2),120).
orientation(br1gt_1,164,121). coords(br1gt_1,61,163,121).  visible(br1gt_1,121). happensAt(walking(br1gt_1),121). orientation(br1gt_2,171,121). coords(br1gt_2,86,207,121).  visible(br1gt_2,121). happensAt(inactive(br1gt_2),121).
orientation(br1gt_1,164,122). coords(br1gt_1,60,163,122).  visible(br1gt_1,122). happensAt(walking(br1gt_1),122). orientation(br1gt_2,171,122). coords(br1gt_2,86,207,122).  visible(br1gt_2,122). happensAt(inactive(br1gt_2),122).
orientation(br1gt_1,164,123). coords(br1gt_1,60,163,123).  visible(br1gt_1,123). happensAt(walking(br1gt_1),123). orientation(br1gt_2,171,123). coords(br1gt_2,86,207,123).  visible(br1gt_2,123). happensAt(inactive(br1gt_2),123).
orientation(br1gt_1,164,124). coords(br1gt_1,59,162,124).  visible(br1gt_1,124). happensAt(walking(br1gt_1),124). orientation(br1gt_2,171,124). coords(br1gt_2,86,207,124).  visible(br1gt_2,124). happensAt(inactive(br1gt_2),124).
orientation(br1gt_1,164,125). coords(br1gt_1,59,162,125).  visible(br1gt_1,125). happensAt(walking(br1gt_1),125). orientation(br1gt_2,171,125). coords(br1gt_2,86,207,125).  visible(br1gt_2,125). happensAt(inactive(br1gt_2),125).
orientation(br1gt_1,164,126). coords(br1gt_1,59,162,126).  visible(br1gt_1,126). happensAt(walking(br1gt_1),126). orientation(br1gt_2,171,126). coords(br1gt_2,86,207,126).  visible(br1gt_2,126). happensAt(inactive(br1gt_2),126).
orientation(br1gt_1,164,127). coords(br1gt_1,59,162,127).  visible(br1gt_1,127). happensAt(walking(br1gt_1),127). orientation(br1gt_2,171,127). coords(br1gt_2,86,207,127).  visible(br1gt_2,127). happensAt(inactive(br1gt_2),127).
orientation(br1gt_1,164,128). coords(br1gt_1,59,162,128).  visible(br1gt_1,128). happensAt(walking(br1gt_1),128). orientation(br1gt_2,171,128). coords(br1gt_2,86,207,128).  visible(br1gt_2,128). happensAt(inactive(br1gt_2),128).
orientation(br1gt_1,164,129). coords(br1gt_1,58,161,129).  visible(br1gt_1,129). happensAt(walking(br1gt_1),129). orientation(br1gt_2,171,129). coords(br1gt_2,86,207,129).  visible(br1gt_2,129). happensAt(inactive(br1gt_2),129).
orientation(br1gt_1,164,130). coords(br1gt_1,58,161,130).  visible(br1gt_1,130). happensAt(walking(br1gt_1),130). orientation(br1gt_2,171,130). coords(br1gt_2,86,207,130).  visible(br1gt_2,130). happensAt(inactive(br1gt_2),130).
orientation(br1gt_1,164,131). coords(br1gt_1,56,161,131).  visible(br1gt_1,131). happensAt(walking(br1gt_1),131). orientation(br1gt_2,171,131). coords(br1gt_2,86,207,131).  visible(br1gt_2,131). happensAt(inactive(br1gt_2),131).
orientation(br1gt_1,164,132). coords(br1gt_1,56,161,132).  visible(br1gt_1,132). happensAt(walking(br1gt_1),132). orientation(br1gt_2,171,132). coords(br1gt_2,86,207,132).  visible(br1gt_2,132). happensAt(inactive(br1gt_2),132).
#show holdsAt/2.