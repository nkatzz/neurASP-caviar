
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

	orientation(wk2gt_1,136,118). coords(wk2gt_1,115,52,118).  visible(wk2gt_1,118). happensAt(inactive(wk2gt_1),118). orientation(wk2gt_3,164,118). coords(wk2gt_3,196,193,118).  visible(wk2gt_3,118). happensAt(walking(wk2gt_3),118).
orientation(wk2gt_1,136,119). coords(wk2gt_1,115,52,119).  visible(wk2gt_1,119). happensAt(inactive(wk2gt_1),119). orientation(wk2gt_3,164,119). coords(wk2gt_3,194,192,119).  visible(wk2gt_3,119). happensAt(walking(wk2gt_3),119).
orientation(wk2gt_1,132,120). coords(wk2gt_1,114,52,120).  visible(wk2gt_1,120). happensAt(inactive(wk2gt_1),120). orientation(wk2gt_3,164,120). coords(wk2gt_3,194,189,120).  visible(wk2gt_3,120). happensAt(walking(wk2gt_3),120).
orientation(wk2gt_1,132,121). coords(wk2gt_1,114,52,121).  visible(wk2gt_1,121). happensAt(inactive(wk2gt_1),121). orientation(wk2gt_3,164,121). coords(wk2gt_3,195,187,121).  visible(wk2gt_3,121). happensAt(walking(wk2gt_3),121).
orientation(wk2gt_1,132,122). coords(wk2gt_1,114,52,122).  visible(wk2gt_1,122). happensAt(inactive(wk2gt_1),122). orientation(wk2gt_3,164,122). coords(wk2gt_3,195,185,122).  visible(wk2gt_3,122). happensAt(walking(wk2gt_3),122).
orientation(wk2gt_1,132,123). coords(wk2gt_1,114,52,123).  visible(wk2gt_1,123). happensAt(inactive(wk2gt_1),123). orientation(wk2gt_3,164,123). coords(wk2gt_3,194,182,123).  visible(wk2gt_3,123). happensAt(walking(wk2gt_3),123).
orientation(wk2gt_1,132,124). coords(wk2gt_1,114,52,124).  visible(wk2gt_1,124). happensAt(inactive(wk2gt_1),124). orientation(wk2gt_3,164,124). coords(wk2gt_3,194,179,124).  visible(wk2gt_3,124). happensAt(walking(wk2gt_3),124).
orientation(wk2gt_1,132,125). coords(wk2gt_1,114,52,125).  visible(wk2gt_1,125). happensAt(inactive(wk2gt_1),125). orientation(wk2gt_3,164,125). coords(wk2gt_3,194,176,125).  visible(wk2gt_3,125). happensAt(walking(wk2gt_3),125).
orientation(wk2gt_1,132,126). coords(wk2gt_1,114,52,126).  visible(wk2gt_1,126). happensAt(inactive(wk2gt_1),126). orientation(wk2gt_3,164,126). coords(wk2gt_3,194,176,126).  visible(wk2gt_3,126). happensAt(walking(wk2gt_3),126).
orientation(wk2gt_1,132,127). coords(wk2gt_1,114,52,127).  visible(wk2gt_1,127). happensAt(inactive(wk2gt_1),127). orientation(wk2gt_3,164,127). coords(wk2gt_3,193,174,127).  visible(wk2gt_3,127). happensAt(walking(wk2gt_3),127).
orientation(wk2gt_1,132,128). coords(wk2gt_1,114,52,128).  visible(wk2gt_1,128). happensAt(inactive(wk2gt_1),128). orientation(wk2gt_3,163,128). coords(wk2gt_3,191,174,128).  visible(wk2gt_3,128). happensAt(active(wk2gt_3),128).
orientation(wk2gt_1,132,129). coords(wk2gt_1,114,52,129).  visible(wk2gt_1,129). happensAt(inactive(wk2gt_1),129). orientation(wk2gt_3,162,129). coords(wk2gt_3,190,172,129).  visible(wk2gt_3,129). happensAt(active(wk2gt_3),129).
orientation(wk2gt_1,132,130). coords(wk2gt_1,114,52,130).  visible(wk2gt_1,130). happensAt(inactive(wk2gt_1),130). orientation(wk2gt_3,157,130). coords(wk2gt_3,189,172,130).  visible(wk2gt_3,130). happensAt(active(wk2gt_3),130).
orientation(wk2gt_1,132,131). coords(wk2gt_1,114,52,131).  visible(wk2gt_1,131). happensAt(inactive(wk2gt_1),131). orientation(wk2gt_3,154,131). coords(wk2gt_3,186,171,131).  visible(wk2gt_3,131). happensAt(active(wk2gt_3),131).
orientation(wk2gt_1,132,132). coords(wk2gt_1,114,52,132).  visible(wk2gt_1,132). happensAt(inactive(wk2gt_1),132). orientation(wk2gt_3,154,132). coords(wk2gt_3,184,169,132).  visible(wk2gt_3,132). happensAt(active(wk2gt_3),132).
orientation(wk2gt_1,132,133). coords(wk2gt_1,114,52,133).  visible(wk2gt_1,133). happensAt(inactive(wk2gt_1),133). orientation(wk2gt_3,154,133). coords(wk2gt_3,182,168,133).  visible(wk2gt_3,133). happensAt(active(wk2gt_3),133).
orientation(wk2gt_1,132,134). coords(wk2gt_1,114,52,134).  visible(wk2gt_1,134). happensAt(inactive(wk2gt_1),134). orientation(wk2gt_3,154,134). coords(wk2gt_3,182,166,134).  visible(wk2gt_3,134). happensAt(active(wk2gt_3),134).
orientation(wk2gt_1,132,135). coords(wk2gt_1,114,52,135).  visible(wk2gt_1,135). happensAt(inactive(wk2gt_1),135). orientation(wk2gt_3,154,135). coords(wk2gt_3,181,164,135).  visible(wk2gt_3,135). happensAt(active(wk2gt_3),135).
orientation(wk2gt_1,132,136). coords(wk2gt_1,114,52,136).  visible(wk2gt_1,136). happensAt(inactive(wk2gt_1),136). orientation(wk2gt_3,154,136). coords(wk2gt_3,181,162,136).  visible(wk2gt_3,136). happensAt(active(wk2gt_3),136).
orientation(wk2gt_1,132,137). coords(wk2gt_1,114,52,137).  visible(wk2gt_1,137). happensAt(inactive(wk2gt_1),137). orientation(wk2gt_3,154,137). coords(wk2gt_3,181,161,137).  visible(wk2gt_3,137). happensAt(active(wk2gt_3),137).
orientation(wk2gt_1,132,138). coords(wk2gt_1,114,52,138).  visible(wk2gt_1,138). happensAt(inactive(wk2gt_1),138). orientation(wk2gt_3,154,138). coords(wk2gt_3,180,159,138).  visible(wk2gt_3,138). happensAt(active(wk2gt_3),138).
orientation(wk2gt_1,132,139). coords(wk2gt_1,114,52,139).  visible(wk2gt_1,139). happensAt(inactive(wk2gt_1),139). orientation(wk2gt_3,154,139). coords(wk2gt_3,180,159,139).  visible(wk2gt_3,139). happensAt(active(wk2gt_3),139).
orientation(wk2gt_1,132,140). coords(wk2gt_1,114,52,140).  visible(wk2gt_1,140). happensAt(inactive(wk2gt_1),140). orientation(wk2gt_3,154,140). coords(wk2gt_3,179,159,140).  visible(wk2gt_3,140). happensAt(active(wk2gt_3),140).
orientation(wk2gt_1,132,141). coords(wk2gt_1,114,52,141).  visible(wk2gt_1,141). happensAt(inactive(wk2gt_1),141). orientation(wk2gt_3,154,141). coords(wk2gt_3,179,159,141).  visible(wk2gt_3,141). happensAt(active(wk2gt_3),141).
orientation(wk2gt_1,132,142). coords(wk2gt_1,114,53,142).  visible(wk2gt_1,142). happensAt(inactive(wk2gt_1),142). orientation(wk2gt_3,154,142). coords(wk2gt_3,179,158,142).  visible(wk2gt_3,142). happensAt(active(wk2gt_3),142).
#show holdsAt/2.