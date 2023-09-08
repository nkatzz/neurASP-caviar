
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

	orientation(mc1gt_1,145,105). coords(mc1gt_1,95,82,105).  visible(mc1gt_1,105). happensAt(walking(mc1gt_1),105). orientation(mc1gt_2,144,105). coords(mc1gt_2,76,68,105).  visible(mc1gt_2,105). happensAt(walking(mc1gt_2),105).
orientation(mc1gt_1,145,106). coords(mc1gt_1,95,82,106).  visible(mc1gt_1,106). happensAt(walking(mc1gt_1),106). orientation(mc1gt_2,144,106). coords(mc1gt_2,76,72,106).  visible(mc1gt_2,106). happensAt(walking(mc1gt_2),106).
orientation(mc1gt_1,145,107). coords(mc1gt_1,95,82,107).  visible(mc1gt_1,107). happensAt(walking(mc1gt_1),107). orientation(mc1gt_2,144,107). coords(mc1gt_2,76,73,107).  visible(mc1gt_2,107). happensAt(walking(mc1gt_2),107).
orientation(mc1gt_1,145,108). coords(mc1gt_1,97,82,108).  visible(mc1gt_1,108). happensAt(walking(mc1gt_1),108). orientation(mc1gt_2,144,108). coords(mc1gt_2,76,73,108).  visible(mc1gt_2,108). happensAt(walking(mc1gt_2),108).
orientation(mc1gt_1,145,109). coords(mc1gt_1,98,82,109).  visible(mc1gt_1,109). happensAt(walking(mc1gt_1),109). orientation(mc1gt_2,144,109). coords(mc1gt_2,76,74,109).  visible(mc1gt_2,109). happensAt(walking(mc1gt_2),109).
orientation(mc1gt_1,145,110). coords(mc1gt_1,99,82,110).  visible(mc1gt_1,110). happensAt(walking(mc1gt_1),110). orientation(mc1gt_2,144,110). coords(mc1gt_2,76,74,110).  visible(mc1gt_2,110). happensAt(walking(mc1gt_2),110).
orientation(mc1gt_1,145,111). coords(mc1gt_1,100,82,111).  visible(mc1gt_1,111). happensAt(walking(mc1gt_1),111). orientation(mc1gt_2,144,111). coords(mc1gt_2,77,74,111).  visible(mc1gt_2,111). happensAt(walking(mc1gt_2),111).
orientation(mc1gt_1,145,112). coords(mc1gt_1,100,82,112).  visible(mc1gt_1,112). happensAt(walking(mc1gt_1),112). orientation(mc1gt_2,144,112). coords(mc1gt_2,77,75,112).  visible(mc1gt_2,112). happensAt(walking(mc1gt_2),112).
orientation(mc1gt_1,145,113). coords(mc1gt_1,101,82,113).  visible(mc1gt_1,113). happensAt(walking(mc1gt_1),113). orientation(mc1gt_2,144,113). coords(mc1gt_2,76,75,113).  visible(mc1gt_2,113). happensAt(walking(mc1gt_2),113).
orientation(mc1gt_1,145,114). coords(mc1gt_1,102,82,114).  visible(mc1gt_1,114). happensAt(walking(mc1gt_1),114). orientation(mc1gt_2,144,114). coords(mc1gt_2,76,76,114).  visible(mc1gt_2,114). happensAt(walking(mc1gt_2),114).
orientation(mc1gt_1,145,115). coords(mc1gt_1,102,82,115).  visible(mc1gt_1,115). happensAt(walking(mc1gt_1),115). orientation(mc1gt_2,144,115). coords(mc1gt_2,76,77,115).  visible(mc1gt_2,115). happensAt(walking(mc1gt_2),115).
orientation(mc1gt_1,145,116). coords(mc1gt_1,102,82,116).  visible(mc1gt_1,116). happensAt(walking(mc1gt_1),116). orientation(mc1gt_2,144,116). coords(mc1gt_2,77,78,116).  visible(mc1gt_2,116). happensAt(walking(mc1gt_2),116).
orientation(mc1gt_1,145,117). coords(mc1gt_1,102,82,117).  visible(mc1gt_1,117). happensAt(walking(mc1gt_1),117). orientation(mc1gt_2,144,117). coords(mc1gt_2,77,78,117).  visible(mc1gt_2,117). happensAt(walking(mc1gt_2),117).
orientation(mc1gt_1,145,118). coords(mc1gt_1,102,82,118).  visible(mc1gt_1,118). happensAt(walking(mc1gt_1),118). orientation(mc1gt_2,144,118). coords(mc1gt_2,77,79,118).  visible(mc1gt_2,118). happensAt(walking(mc1gt_2),118).
orientation(mc1gt_1,145,119). coords(mc1gt_1,102,82,119).  visible(mc1gt_1,119). happensAt(walking(mc1gt_1),119). orientation(mc1gt_2,144,119). coords(mc1gt_2,77,79,119).  visible(mc1gt_2,119). happensAt(walking(mc1gt_2),119).
orientation(mc1gt_1,145,120). coords(mc1gt_1,103,82,120).  visible(mc1gt_1,120). happensAt(walking(mc1gt_1),120). orientation(mc1gt_2,144,120). coords(mc1gt_2,77,79,120).  visible(mc1gt_2,120). happensAt(walking(mc1gt_2),120).
orientation(mc1gt_1,145,121). coords(mc1gt_1,103,82,121).  visible(mc1gt_1,121). happensAt(walking(mc1gt_1),121). orientation(mc1gt_2,144,121). coords(mc1gt_2,77,80,121).  visible(mc1gt_2,121). happensAt(walking(mc1gt_2),121).
orientation(mc1gt_1,145,122). coords(mc1gt_1,105,82,122).  visible(mc1gt_1,122). happensAt(walking(mc1gt_1),122). orientation(mc1gt_2,144,122). coords(mc1gt_2,77,80,122).  visible(mc1gt_2,122). happensAt(walking(mc1gt_2),122).
orientation(mc1gt_1,145,123). coords(mc1gt_1,106,82,123).  visible(mc1gt_1,123). happensAt(walking(mc1gt_1),123). orientation(mc1gt_2,144,123). coords(mc1gt_2,77,81,123).  visible(mc1gt_2,123). happensAt(walking(mc1gt_2),123).
orientation(mc1gt_1,145,124). coords(mc1gt_1,107,82,124).  visible(mc1gt_1,124). happensAt(walking(mc1gt_1),124). orientation(mc1gt_2,144,124). coords(mc1gt_2,77,83,124).  visible(mc1gt_2,124). happensAt(walking(mc1gt_2),124).
orientation(mc1gt_1,145,125). coords(mc1gt_1,108,82,125).  visible(mc1gt_1,125). happensAt(walking(mc1gt_1),125). orientation(mc1gt_2,144,125). coords(mc1gt_2,76,85,125).  visible(mc1gt_2,125). happensAt(walking(mc1gt_2),125).
orientation(mc1gt_1,145,126). coords(mc1gt_1,109,82,126).  visible(mc1gt_1,126). happensAt(walking(mc1gt_1),126). orientation(mc1gt_2,144,126). coords(mc1gt_2,76,86,126).  visible(mc1gt_2,126). happensAt(walking(mc1gt_2),126).
orientation(mc1gt_1,145,127). coords(mc1gt_1,110,82,127).  visible(mc1gt_1,127). happensAt(walking(mc1gt_1),127). orientation(mc1gt_2,144,127). coords(mc1gt_2,76,86,127).  visible(mc1gt_2,127). happensAt(walking(mc1gt_2),127).
orientation(mc1gt_1,145,128). coords(mc1gt_1,110,82,128).  visible(mc1gt_1,128). happensAt(walking(mc1gt_1),128). orientation(mc1gt_2,144,128). coords(mc1gt_2,76,87,128).  visible(mc1gt_2,128). happensAt(walking(mc1gt_2),128).
orientation(mc1gt_1,145,129). coords(mc1gt_1,110,82,129).  visible(mc1gt_1,129). happensAt(walking(mc1gt_1),129). orientation(mc1gt_2,144,129). coords(mc1gt_2,76,87,129).  visible(mc1gt_2,129). happensAt(walking(mc1gt_2),129).
#show holdsAt/2.