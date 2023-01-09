
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

	orientation(wk2gt_1,137,108). coords(wk2gt_1,115,52,108).  visible(wk2gt_1,108). happensAt(inactive(wk2gt_1),108). orientation(wk2gt_2,141,108). coords(wk2gt_2,98,62,108).  visible(wk2gt_2,108). happensAt(inactive(wk2gt_2),108).
orientation(wk2gt_1,134,109). coords(wk2gt_1,115,52,109).  visible(wk2gt_1,109). happensAt(inactive(wk2gt_1),109). orientation(wk2gt_2,141,109). coords(wk2gt_2,98,62,109).  visible(wk2gt_2,109). happensAt(inactive(wk2gt_2),109).
orientation(wk2gt_1,136,110). coords(wk2gt_1,115,52,110).  visible(wk2gt_1,110). happensAt(inactive(wk2gt_1),110). orientation(wk2gt_2,141,110). coords(wk2gt_2,98,62,110).  visible(wk2gt_2,110). happensAt(inactive(wk2gt_2),110).
orientation(wk2gt_1,134,111). coords(wk2gt_1,115,52,111).  visible(wk2gt_1,111). happensAt(inactive(wk2gt_1),111). orientation(wk2gt_2,141,111). coords(wk2gt_2,98,62,111).  visible(wk2gt_2,111). happensAt(inactive(wk2gt_2),111).
orientation(wk2gt_1,136,112). coords(wk2gt_1,115,52,112).  visible(wk2gt_1,112). happensAt(inactive(wk2gt_1),112). orientation(wk2gt_2,141,112). coords(wk2gt_2,98,62,112).  visible(wk2gt_2,112). happensAt(inactive(wk2gt_2),112).
orientation(wk2gt_1,139,113). coords(wk2gt_1,115,52,113).  visible(wk2gt_1,113). happensAt(inactive(wk2gt_1),113). orientation(wk2gt_2,141,113). coords(wk2gt_2,98,62,113).  visible(wk2gt_2,113). happensAt(inactive(wk2gt_2),113).
orientation(wk2gt_1,136,114). coords(wk2gt_1,115,52,114).  visible(wk2gt_1,114). happensAt(inactive(wk2gt_1),114). orientation(wk2gt_2,141,114). coords(wk2gt_2,98,62,114).  visible(wk2gt_2,114). happensAt(inactive(wk2gt_2),114).
orientation(wk2gt_1,135,115). coords(wk2gt_1,115,52,115).  visible(wk2gt_1,115). happensAt(inactive(wk2gt_1),115). orientation(wk2gt_2,141,115). coords(wk2gt_2,98,62,115).  visible(wk2gt_2,115). happensAt(inactive(wk2gt_2),115).
orientation(wk2gt_1,136,116). coords(wk2gt_1,115,52,116).  visible(wk2gt_1,116). happensAt(inactive(wk2gt_1),116). orientation(wk2gt_2,141,116). coords(wk2gt_2,98,62,116).  visible(wk2gt_2,116). happensAt(inactive(wk2gt_2),116).
orientation(wk2gt_1,135,117). coords(wk2gt_1,115,52,117).  visible(wk2gt_1,117). happensAt(inactive(wk2gt_1),117). orientation(wk2gt_2,141,117). coords(wk2gt_2,98,62,117).  visible(wk2gt_2,117). happensAt(inactive(wk2gt_2),117).
orientation(wk2gt_1,136,118). coords(wk2gt_1,115,52,118).  visible(wk2gt_1,118). happensAt(inactive(wk2gt_1),118). orientation(wk2gt_2,141,118). coords(wk2gt_2,98,62,118).  visible(wk2gt_2,118). happensAt(inactive(wk2gt_2),118).
orientation(wk2gt_1,136,119). coords(wk2gt_1,115,52,119).  visible(wk2gt_1,119). happensAt(inactive(wk2gt_1),119). orientation(wk2gt_2,141,119). coords(wk2gt_2,98,62,119).  visible(wk2gt_2,119). happensAt(inactive(wk2gt_2),119).
orientation(wk2gt_1,132,120). coords(wk2gt_1,114,52,120).  visible(wk2gt_1,120). happensAt(inactive(wk2gt_1),120). orientation(wk2gt_2,141,120). coords(wk2gt_2,98,62,120).  visible(wk2gt_2,120). happensAt(inactive(wk2gt_2),120).
orientation(wk2gt_1,132,121). coords(wk2gt_1,114,52,121).  visible(wk2gt_1,121). happensAt(inactive(wk2gt_1),121). orientation(wk2gt_2,141,121). coords(wk2gt_2,98,62,121).  visible(wk2gt_2,121). happensAt(inactive(wk2gt_2),121).
orientation(wk2gt_1,132,122). coords(wk2gt_1,114,52,122).  visible(wk2gt_1,122). happensAt(inactive(wk2gt_1),122). orientation(wk2gt_2,141,122). coords(wk2gt_2,98,62,122).  visible(wk2gt_2,122). happensAt(inactive(wk2gt_2),122).
orientation(wk2gt_1,132,123). coords(wk2gt_1,114,52,123).  visible(wk2gt_1,123). happensAt(inactive(wk2gt_1),123). orientation(wk2gt_2,141,123). coords(wk2gt_2,98,62,123).  visible(wk2gt_2,123). happensAt(inactive(wk2gt_2),123).
orientation(wk2gt_1,132,124). coords(wk2gt_1,114,52,124).  visible(wk2gt_1,124). happensAt(inactive(wk2gt_1),124). orientation(wk2gt_2,141,124). coords(wk2gt_2,98,62,124).  visible(wk2gt_2,124). happensAt(inactive(wk2gt_2),124).
orientation(wk2gt_1,132,125). coords(wk2gt_1,114,52,125).  visible(wk2gt_1,125). happensAt(inactive(wk2gt_1),125). orientation(wk2gt_2,141,125). coords(wk2gt_2,98,62,125).  visible(wk2gt_2,125). happensAt(inactive(wk2gt_2),125).
orientation(wk2gt_1,132,126). coords(wk2gt_1,114,52,126).  visible(wk2gt_1,126). happensAt(inactive(wk2gt_1),126). orientation(wk2gt_2,141,126). coords(wk2gt_2,98,62,126).  visible(wk2gt_2,126). happensAt(inactive(wk2gt_2),126).
orientation(wk2gt_1,132,127). coords(wk2gt_1,114,52,127).  visible(wk2gt_1,127). happensAt(inactive(wk2gt_1),127). orientation(wk2gt_2,141,127). coords(wk2gt_2,98,62,127).  visible(wk2gt_2,127). happensAt(inactive(wk2gt_2),127).
orientation(wk2gt_1,132,128). coords(wk2gt_1,114,52,128).  visible(wk2gt_1,128). happensAt(inactive(wk2gt_1),128). orientation(wk2gt_2,141,128). coords(wk2gt_2,98,62,128).  visible(wk2gt_2,128). happensAt(inactive(wk2gt_2),128).
orientation(wk2gt_1,132,129). coords(wk2gt_1,114,52,129).  visible(wk2gt_1,129). happensAt(inactive(wk2gt_1),129). orientation(wk2gt_2,141,129). coords(wk2gt_2,98,62,129).  visible(wk2gt_2,129). happensAt(inactive(wk2gt_2),129).
orientation(wk2gt_1,132,130). coords(wk2gt_1,114,52,130).  visible(wk2gt_1,130). happensAt(inactive(wk2gt_1),130). orientation(wk2gt_2,141,130). coords(wk2gt_2,98,62,130).  visible(wk2gt_2,130). happensAt(inactive(wk2gt_2),130).
orientation(wk2gt_1,132,131). coords(wk2gt_1,114,52,131).  visible(wk2gt_1,131). happensAt(inactive(wk2gt_1),131). orientation(wk2gt_2,141,131). coords(wk2gt_2,98,62,131).  visible(wk2gt_2,131). happensAt(inactive(wk2gt_2),131).
orientation(wk2gt_1,132,132). coords(wk2gt_1,114,52,132).  visible(wk2gt_1,132). happensAt(inactive(wk2gt_1),132). orientation(wk2gt_2,141,132). coords(wk2gt_2,98,62,132).  visible(wk2gt_2,132). happensAt(inactive(wk2gt_2),132).
#show holdsAt/2.