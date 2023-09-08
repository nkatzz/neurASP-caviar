
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

	orientation(wk1gt_4,0,111). coords(wk1gt_4,44,187,111).  visible(wk1gt_4,111). happensAt(walking(wk1gt_4),111). orientation(wk1gt_5,0,111). coords(wk1gt_5,39,179,111).  visible(wk1gt_5,111). happensAt(walking(wk1gt_5),111).
orientation(wk1gt_4,0,112). coords(wk1gt_4,44,187,112).  visible(wk1gt_4,112). happensAt(walking(wk1gt_4),112). orientation(wk1gt_5,0,112). coords(wk1gt_5,39,179,112).  visible(wk1gt_5,112). happensAt(walking(wk1gt_5),112).
orientation(wk1gt_4,0,113). coords(wk1gt_4,44,187,113).  visible(wk1gt_4,113). happensAt(walking(wk1gt_4),113). orientation(wk1gt_5,174,113). coords(wk1gt_5,39,178,113).  visible(wk1gt_5,113). happensAt(walking(wk1gt_5),113).
orientation(wk1gt_4,0,114). coords(wk1gt_4,44,187,114).  visible(wk1gt_4,114). happensAt(walking(wk1gt_4),114). orientation(wk1gt_5,174,114). coords(wk1gt_5,40,177,114).  visible(wk1gt_5,114). happensAt(walking(wk1gt_5),114).
orientation(wk1gt_4,0,115). coords(wk1gt_4,44,187,115).  visible(wk1gt_4,115). happensAt(walking(wk1gt_4),115). orientation(wk1gt_5,174,115). coords(wk1gt_5,41,176,115).  visible(wk1gt_5,115). happensAt(walking(wk1gt_5),115).
orientation(wk1gt_4,0,116). coords(wk1gt_4,44,187,116).  visible(wk1gt_4,116). happensAt(walking(wk1gt_4),116). orientation(wk1gt_5,174,116). coords(wk1gt_5,41,176,116).  visible(wk1gt_5,116). happensAt(walking(wk1gt_5),116).
orientation(wk1gt_4,0,117). coords(wk1gt_4,44,187,117).  visible(wk1gt_4,117). happensAt(walking(wk1gt_4),117). orientation(wk1gt_5,169,117). coords(wk1gt_5,41,176,117).  visible(wk1gt_5,117). happensAt(walking(wk1gt_5),117).
orientation(wk1gt_4,0,118). coords(wk1gt_4,44,187,118).  visible(wk1gt_4,118). happensAt(walking(wk1gt_4),118). orientation(wk1gt_5,169,118). coords(wk1gt_5,40,176,118).  visible(wk1gt_5,118). happensAt(walking(wk1gt_5),118).
orientation(wk1gt_4,0,119). coords(wk1gt_4,44,187,119).  visible(wk1gt_4,119). happensAt(walking(wk1gt_4),119). orientation(wk1gt_5,169,119). coords(wk1gt_5,40,176,119).  visible(wk1gt_5,119). happensAt(walking(wk1gt_5),119).
orientation(wk1gt_4,0,120). coords(wk1gt_4,44,187,120).  visible(wk1gt_4,120). happensAt(walking(wk1gt_4),120). orientation(wk1gt_5,169,120). coords(wk1gt_5,40,175,120).  visible(wk1gt_5,120). happensAt(walking(wk1gt_5),120).
orientation(wk1gt_4,0,121). coords(wk1gt_4,45,186,121).  visible(wk1gt_4,121). happensAt(walking(wk1gt_4),121). orientation(wk1gt_5,169,121). coords(wk1gt_5,40,175,121).  visible(wk1gt_5,121). happensAt(walking(wk1gt_5),121).
orientation(wk1gt_4,0,122). coords(wk1gt_4,45,186,122).  visible(wk1gt_4,122). happensAt(walking(wk1gt_4),122). orientation(wk1gt_5,169,122). coords(wk1gt_5,40,174,122).  visible(wk1gt_5,122). happensAt(walking(wk1gt_5),122).
orientation(wk1gt_4,0,123). coords(wk1gt_4,45,186,123).  visible(wk1gt_4,123). happensAt(walking(wk1gt_4),123). orientation(wk1gt_5,169,123). coords(wk1gt_5,40,173,123).  visible(wk1gt_5,123). happensAt(walking(wk1gt_5),123).
orientation(wk1gt_4,0,124). coords(wk1gt_4,45,186,124).  visible(wk1gt_4,124). happensAt(walking(wk1gt_4),124). orientation(wk1gt_5,169,124). coords(wk1gt_5,40,173,124).  visible(wk1gt_5,124). happensAt(walking(wk1gt_5),124).
orientation(wk1gt_4,0,125). coords(wk1gt_4,45,186,125).  visible(wk1gt_4,125). happensAt(walking(wk1gt_4),125). orientation(wk1gt_5,169,125). coords(wk1gt_5,40,172,125).  visible(wk1gt_5,125). happensAt(walking(wk1gt_5),125).
orientation(wk1gt_4,0,126). coords(wk1gt_4,45,186,126).  visible(wk1gt_4,126). happensAt(walking(wk1gt_4),126). orientation(wk1gt_5,169,126). coords(wk1gt_5,42,172,126).  visible(wk1gt_5,126). happensAt(walking(wk1gt_5),126).
orientation(wk1gt_4,0,127). coords(wk1gt_4,45,186,127).  visible(wk1gt_4,127). happensAt(walking(wk1gt_4),127). orientation(wk1gt_5,169,127). coords(wk1gt_5,42,172,127).  visible(wk1gt_5,127). happensAt(walking(wk1gt_5),127).
orientation(wk1gt_4,0,128). coords(wk1gt_4,45,186,128).  visible(wk1gt_4,128). happensAt(walking(wk1gt_4),128). orientation(wk1gt_5,169,128). coords(wk1gt_5,42,172,128).  visible(wk1gt_5,128). happensAt(walking(wk1gt_5),128).
orientation(wk1gt_4,0,129). coords(wk1gt_4,45,186,129).  visible(wk1gt_4,129). happensAt(walking(wk1gt_4),129). orientation(wk1gt_5,169,129). coords(wk1gt_5,43,172,129).  visible(wk1gt_5,129). happensAt(walking(wk1gt_5),129).
orientation(wk1gt_4,0,130). coords(wk1gt_4,45,186,130).  visible(wk1gt_4,130). happensAt(walking(wk1gt_4),130). orientation(wk1gt_5,169,130). coords(wk1gt_5,43,171,130).  visible(wk1gt_5,130). happensAt(walking(wk1gt_5),130).
orientation(wk1gt_4,0,131). coords(wk1gt_4,45,186,131).  visible(wk1gt_4,131). happensAt(walking(wk1gt_4),131). orientation(wk1gt_5,169,131). coords(wk1gt_5,43,171,131).  visible(wk1gt_5,131). happensAt(walking(wk1gt_5),131).
orientation(wk1gt_4,0,132). coords(wk1gt_4,45,186,132).  visible(wk1gt_4,132). happensAt(walking(wk1gt_4),132). orientation(wk1gt_5,169,132). coords(wk1gt_5,43,171,132).  visible(wk1gt_5,132). happensAt(walking(wk1gt_5),132).
orientation(wk1gt_4,0,133). coords(wk1gt_4,45,185,133).  visible(wk1gt_4,133). happensAt(walking(wk1gt_4),133). orientation(wk1gt_5,169,133). coords(wk1gt_5,44,170,133).  visible(wk1gt_5,133). happensAt(walking(wk1gt_5),133).
orientation(wk1gt_4,0,134). coords(wk1gt_4,45,185,134).  visible(wk1gt_4,134). happensAt(walking(wk1gt_4),134). orientation(wk1gt_5,169,134). coords(wk1gt_5,44,169,134).  visible(wk1gt_5,134). happensAt(walking(wk1gt_5),134).
orientation(wk1gt_4,0,135). coords(wk1gt_4,45,185,135).  visible(wk1gt_4,135). happensAt(walking(wk1gt_4),135). orientation(wk1gt_5,169,135). coords(wk1gt_5,44,169,135).  visible(wk1gt_5,135). happensAt(walking(wk1gt_5),135).
#show holdsAt/2.