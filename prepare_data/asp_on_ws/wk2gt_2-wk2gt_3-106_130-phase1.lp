
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

	orientation(wk2gt_2,141,106). coords(wk2gt_2,98,62,106).  visible(wk2gt_2,106). happensAt(inactive(wk2gt_2),106). orientation(wk2gt_3,167,106). coords(wk2gt_3,207,226,106).  visible(wk2gt_3,106). happensAt(walking(wk2gt_3),106).
orientation(wk2gt_2,141,107). coords(wk2gt_2,98,62,107).  visible(wk2gt_2,107). happensAt(inactive(wk2gt_2),107). orientation(wk2gt_3,167,107). coords(wk2gt_3,205,222,107).  visible(wk2gt_3,107). happensAt(walking(wk2gt_3),107).
orientation(wk2gt_2,141,108). coords(wk2gt_2,98,62,108).  visible(wk2gt_2,108). happensAt(inactive(wk2gt_2),108). orientation(wk2gt_3,167,108). coords(wk2gt_3,203,218,108).  visible(wk2gt_3,108). happensAt(walking(wk2gt_3),108).
orientation(wk2gt_2,141,109). coords(wk2gt_2,98,62,109).  visible(wk2gt_2,109). happensAt(inactive(wk2gt_2),109). orientation(wk2gt_3,168,109). coords(wk2gt_3,201,212,109).  visible(wk2gt_3,109). happensAt(walking(wk2gt_3),109).
orientation(wk2gt_2,141,110). coords(wk2gt_2,98,62,110).  visible(wk2gt_2,110). happensAt(inactive(wk2gt_2),110). orientation(wk2gt_3,168,110). coords(wk2gt_3,200,210,110).  visible(wk2gt_3,110). happensAt(walking(wk2gt_3),110).
orientation(wk2gt_2,141,111). coords(wk2gt_2,98,62,111).  visible(wk2gt_2,111). happensAt(inactive(wk2gt_2),111). orientation(wk2gt_3,168,111). coords(wk2gt_3,200,207,111).  visible(wk2gt_3,111). happensAt(walking(wk2gt_3),111).
orientation(wk2gt_2,141,112). coords(wk2gt_2,98,62,112).  visible(wk2gt_2,112). happensAt(inactive(wk2gt_2),112). orientation(wk2gt_3,168,112). coords(wk2gt_3,200,205,112).  visible(wk2gt_3,112). happensAt(walking(wk2gt_3),112).
orientation(wk2gt_2,141,113). coords(wk2gt_2,98,62,113).  visible(wk2gt_2,113). happensAt(inactive(wk2gt_2),113). orientation(wk2gt_3,168,113). coords(wk2gt_3,199,203,113).  visible(wk2gt_3,113). happensAt(walking(wk2gt_3),113).
orientation(wk2gt_2,141,114). coords(wk2gt_2,98,62,114).  visible(wk2gt_2,114). happensAt(inactive(wk2gt_2),114). orientation(wk2gt_3,168,114). coords(wk2gt_3,199,201,114).  visible(wk2gt_3,114). happensAt(walking(wk2gt_3),114).
orientation(wk2gt_2,141,115). coords(wk2gt_2,98,62,115).  visible(wk2gt_2,115). happensAt(inactive(wk2gt_2),115). orientation(wk2gt_3,168,115). coords(wk2gt_3,199,200,115).  visible(wk2gt_3,115). happensAt(walking(wk2gt_3),115).
orientation(wk2gt_2,141,116). coords(wk2gt_2,98,62,116).  visible(wk2gt_2,116). happensAt(inactive(wk2gt_2),116). orientation(wk2gt_3,168,116). coords(wk2gt_3,197,198,116).  visible(wk2gt_3,116). happensAt(walking(wk2gt_3),116).
orientation(wk2gt_2,141,117). coords(wk2gt_2,98,62,117).  visible(wk2gt_2,117). happensAt(inactive(wk2gt_2),117). orientation(wk2gt_3,168,117). coords(wk2gt_3,197,194,117).  visible(wk2gt_3,117). happensAt(walking(wk2gt_3),117).
orientation(wk2gt_2,141,118). coords(wk2gt_2,98,62,118).  visible(wk2gt_2,118). happensAt(inactive(wk2gt_2),118). orientation(wk2gt_3,164,118). coords(wk2gt_3,196,193,118).  visible(wk2gt_3,118). happensAt(walking(wk2gt_3),118).
orientation(wk2gt_2,141,119). coords(wk2gt_2,98,62,119).  visible(wk2gt_2,119). happensAt(inactive(wk2gt_2),119). orientation(wk2gt_3,164,119). coords(wk2gt_3,194,192,119).  visible(wk2gt_3,119). happensAt(walking(wk2gt_3),119).
orientation(wk2gt_2,141,120). coords(wk2gt_2,98,62,120).  visible(wk2gt_2,120). happensAt(inactive(wk2gt_2),120). orientation(wk2gt_3,164,120). coords(wk2gt_3,194,189,120).  visible(wk2gt_3,120). happensAt(walking(wk2gt_3),120).
orientation(wk2gt_2,141,121). coords(wk2gt_2,98,62,121).  visible(wk2gt_2,121). happensAt(inactive(wk2gt_2),121). orientation(wk2gt_3,164,121). coords(wk2gt_3,195,187,121).  visible(wk2gt_3,121). happensAt(walking(wk2gt_3),121).
orientation(wk2gt_2,141,122). coords(wk2gt_2,98,62,122).  visible(wk2gt_2,122). happensAt(inactive(wk2gt_2),122). orientation(wk2gt_3,164,122). coords(wk2gt_3,195,185,122).  visible(wk2gt_3,122). happensAt(walking(wk2gt_3),122).
orientation(wk2gt_2,141,123). coords(wk2gt_2,98,62,123).  visible(wk2gt_2,123). happensAt(inactive(wk2gt_2),123). orientation(wk2gt_3,164,123). coords(wk2gt_3,194,182,123).  visible(wk2gt_3,123). happensAt(walking(wk2gt_3),123).
orientation(wk2gt_2,141,124). coords(wk2gt_2,98,62,124).  visible(wk2gt_2,124). happensAt(inactive(wk2gt_2),124). orientation(wk2gt_3,164,124). coords(wk2gt_3,194,179,124).  visible(wk2gt_3,124). happensAt(walking(wk2gt_3),124).
orientation(wk2gt_2,141,125). coords(wk2gt_2,98,62,125).  visible(wk2gt_2,125). happensAt(inactive(wk2gt_2),125). orientation(wk2gt_3,164,125). coords(wk2gt_3,194,176,125).  visible(wk2gt_3,125). happensAt(walking(wk2gt_3),125).
orientation(wk2gt_2,141,126). coords(wk2gt_2,98,62,126).  visible(wk2gt_2,126). happensAt(inactive(wk2gt_2),126). orientation(wk2gt_3,164,126). coords(wk2gt_3,194,176,126).  visible(wk2gt_3,126). happensAt(walking(wk2gt_3),126).
orientation(wk2gt_2,141,127). coords(wk2gt_2,98,62,127).  visible(wk2gt_2,127). happensAt(inactive(wk2gt_2),127). orientation(wk2gt_3,164,127). coords(wk2gt_3,193,174,127).  visible(wk2gt_3,127). happensAt(walking(wk2gt_3),127).
orientation(wk2gt_2,141,128). coords(wk2gt_2,98,62,128).  visible(wk2gt_2,128). happensAt(inactive(wk2gt_2),128). orientation(wk2gt_3,163,128). coords(wk2gt_3,191,174,128).  visible(wk2gt_3,128). happensAt(active(wk2gt_3),128).
orientation(wk2gt_2,141,129). coords(wk2gt_2,98,62,129).  visible(wk2gt_2,129). happensAt(inactive(wk2gt_2),129). orientation(wk2gt_3,162,129). coords(wk2gt_3,190,172,129).  visible(wk2gt_3,129). happensAt(active(wk2gt_3),129).
orientation(wk2gt_2,141,130). coords(wk2gt_2,98,62,130).  visible(wk2gt_2,130). happensAt(inactive(wk2gt_2),130). orientation(wk2gt_3,157,130). coords(wk2gt_3,189,172,130).  visible(wk2gt_3,130). happensAt(active(wk2gt_3),130).
#show holdsAt/2.