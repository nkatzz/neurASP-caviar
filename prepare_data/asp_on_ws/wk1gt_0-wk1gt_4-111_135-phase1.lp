
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

	orientation(wk1gt_0,131,111). coords(wk1gt_0,204,124,111).  visible(wk1gt_0,111). happensAt(active(wk1gt_0),111). orientation(wk1gt_4,0,111). coords(wk1gt_4,44,187,111).  visible(wk1gt_4,111). happensAt(walking(wk1gt_4),111).
orientation(wk1gt_0,131,112). coords(wk1gt_0,204,124,112).  visible(wk1gt_0,112). happensAt(inactive(wk1gt_0),112). orientation(wk1gt_4,0,112). coords(wk1gt_4,44,187,112).  visible(wk1gt_4,112). happensAt(walking(wk1gt_4),112).
orientation(wk1gt_0,131,113). coords(wk1gt_0,204,124,113).  visible(wk1gt_0,113). happensAt(inactive(wk1gt_0),113). orientation(wk1gt_4,0,113). coords(wk1gt_4,44,187,113).  visible(wk1gt_4,113). happensAt(walking(wk1gt_4),113).
orientation(wk1gt_0,131,114). coords(wk1gt_0,205,124,114).  visible(wk1gt_0,114). happensAt(inactive(wk1gt_0),114). orientation(wk1gt_4,0,114). coords(wk1gt_4,44,187,114).  visible(wk1gt_4,114). happensAt(walking(wk1gt_4),114).
orientation(wk1gt_0,131,115). coords(wk1gt_0,205,125,115).  visible(wk1gt_0,115). happensAt(inactive(wk1gt_0),115). orientation(wk1gt_4,0,115). coords(wk1gt_4,44,187,115).  visible(wk1gt_4,115). happensAt(walking(wk1gt_4),115).
orientation(wk1gt_0,131,116). coords(wk1gt_0,206,125,116).  visible(wk1gt_0,116). happensAt(inactive(wk1gt_0),116). orientation(wk1gt_4,0,116). coords(wk1gt_4,44,187,116).  visible(wk1gt_4,116). happensAt(walking(wk1gt_4),116).
orientation(wk1gt_0,131,117). coords(wk1gt_0,207,125,117).  visible(wk1gt_0,117). happensAt(inactive(wk1gt_0),117). orientation(wk1gt_4,0,117). coords(wk1gt_4,44,187,117).  visible(wk1gt_4,117). happensAt(walking(wk1gt_4),117).
orientation(wk1gt_0,131,118). coords(wk1gt_0,207,126,118).  visible(wk1gt_0,118). happensAt(active(wk1gt_0),118). orientation(wk1gt_4,0,118). coords(wk1gt_4,44,187,118).  visible(wk1gt_4,118). happensAt(walking(wk1gt_4),118).
orientation(wk1gt_0,131,119). coords(wk1gt_0,206,126,119).  visible(wk1gt_0,119). happensAt(active(wk1gt_0),119). orientation(wk1gt_4,0,119). coords(wk1gt_4,44,187,119).  visible(wk1gt_4,119). happensAt(walking(wk1gt_4),119).
orientation(wk1gt_0,131,120). coords(wk1gt_0,208,128,120).  visible(wk1gt_0,120). happensAt(active(wk1gt_0),120). orientation(wk1gt_4,0,120). coords(wk1gt_4,44,187,120).  visible(wk1gt_4,120). happensAt(walking(wk1gt_4),120).
orientation(wk1gt_0,131,121). coords(wk1gt_0,210,129,121).  visible(wk1gt_0,121). happensAt(active(wk1gt_0),121). orientation(wk1gt_4,0,121). coords(wk1gt_4,45,186,121).  visible(wk1gt_4,121). happensAt(walking(wk1gt_4),121).
orientation(wk1gt_0,131,122). coords(wk1gt_0,210,131,122).  visible(wk1gt_0,122). happensAt(active(wk1gt_0),122). orientation(wk1gt_4,0,122). coords(wk1gt_4,45,186,122).  visible(wk1gt_4,122). happensAt(walking(wk1gt_4),122).
orientation(wk1gt_0,131,123). coords(wk1gt_0,212,133,123).  visible(wk1gt_0,123). happensAt(active(wk1gt_0),123). orientation(wk1gt_4,0,123). coords(wk1gt_4,45,186,123).  visible(wk1gt_4,123). happensAt(walking(wk1gt_4),123).
orientation(wk1gt_0,127,124). coords(wk1gt_0,213,135,124).  visible(wk1gt_0,124). happensAt(active(wk1gt_0),124). orientation(wk1gt_4,0,124). coords(wk1gt_4,45,186,124).  visible(wk1gt_4,124). happensAt(walking(wk1gt_4),124).
orientation(wk1gt_0,131,125). coords(wk1gt_0,214,138,125).  visible(wk1gt_0,125). happensAt(active(wk1gt_0),125). orientation(wk1gt_4,0,125). coords(wk1gt_4,45,186,125).  visible(wk1gt_4,125). happensAt(walking(wk1gt_4),125).
orientation(wk1gt_0,129,126). coords(wk1gt_0,216,140,126).  visible(wk1gt_0,126). happensAt(active(wk1gt_0),126). orientation(wk1gt_4,0,126). coords(wk1gt_4,45,186,126).  visible(wk1gt_4,126). happensAt(walking(wk1gt_4),126).
orientation(wk1gt_0,129,127). coords(wk1gt_0,218,144,127).  visible(wk1gt_0,127). happensAt(walking(wk1gt_0),127). orientation(wk1gt_4,0,127). coords(wk1gt_4,45,186,127).  visible(wk1gt_4,127). happensAt(walking(wk1gt_4),127).
orientation(wk1gt_0,125,128). coords(wk1gt_0,220,146,128).  visible(wk1gt_0,128). happensAt(walking(wk1gt_0),128). orientation(wk1gt_4,0,128). coords(wk1gt_4,45,186,128).  visible(wk1gt_4,128). happensAt(walking(wk1gt_4),128).
orientation(wk1gt_0,125,129). coords(wk1gt_0,222,149,129).  visible(wk1gt_0,129). happensAt(walking(wk1gt_0),129). orientation(wk1gt_4,0,129). coords(wk1gt_4,45,186,129).  visible(wk1gt_4,129). happensAt(walking(wk1gt_4),129).
orientation(wk1gt_0,125,130). coords(wk1gt_0,224,150,130).  visible(wk1gt_0,130). happensAt(walking(wk1gt_0),130). orientation(wk1gt_4,0,130). coords(wk1gt_4,45,186,130).  visible(wk1gt_4,130). happensAt(walking(wk1gt_4),130).
orientation(wk1gt_0,135,131). coords(wk1gt_0,225,152,131).  visible(wk1gt_0,131). happensAt(walking(wk1gt_0),131). orientation(wk1gt_4,0,131). coords(wk1gt_4,45,186,131).  visible(wk1gt_4,131). happensAt(walking(wk1gt_4),131).
orientation(wk1gt_0,135,132). coords(wk1gt_0,226,153,132).  visible(wk1gt_0,132). happensAt(walking(wk1gt_0),132). orientation(wk1gt_4,0,132). coords(wk1gt_4,45,186,132).  visible(wk1gt_4,132). happensAt(walking(wk1gt_4),132).
orientation(wk1gt_0,144,133). coords(wk1gt_0,227,153,133).  visible(wk1gt_0,133). happensAt(walking(wk1gt_0),133). orientation(wk1gt_4,0,133). coords(wk1gt_4,45,185,133).  visible(wk1gt_4,133). happensAt(walking(wk1gt_4),133).
orientation(wk1gt_0,155,134). coords(wk1gt_0,228,154,134).  visible(wk1gt_0,134). happensAt(walking(wk1gt_0),134). orientation(wk1gt_4,0,134). coords(wk1gt_4,45,185,134).  visible(wk1gt_4,134). happensAt(walking(wk1gt_4),134).
orientation(wk1gt_0,165,135). coords(wk1gt_0,229,156,135).  visible(wk1gt_0,135). happensAt(walking(wk1gt_0),135). orientation(wk1gt_4,0,135). coords(wk1gt_4,45,185,135).  visible(wk1gt_4,135). happensAt(walking(wk1gt_4),135).
#show holdsAt/2.