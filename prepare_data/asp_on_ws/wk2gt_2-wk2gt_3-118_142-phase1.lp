
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
orientation(wk2gt_2,141,131). coords(wk2gt_2,98,62,131).  visible(wk2gt_2,131). happensAt(inactive(wk2gt_2),131). orientation(wk2gt_3,154,131). coords(wk2gt_3,186,171,131).  visible(wk2gt_3,131). happensAt(active(wk2gt_3),131).
orientation(wk2gt_2,141,132). coords(wk2gt_2,98,62,132).  visible(wk2gt_2,132). happensAt(inactive(wk2gt_2),132). orientation(wk2gt_3,154,132). coords(wk2gt_3,184,169,132).  visible(wk2gt_3,132). happensAt(active(wk2gt_3),132).
orientation(wk2gt_2,141,133). coords(wk2gt_2,98,62,133).  visible(wk2gt_2,133). happensAt(inactive(wk2gt_2),133). orientation(wk2gt_3,154,133). coords(wk2gt_3,182,168,133).  visible(wk2gt_3,133). happensAt(active(wk2gt_3),133).
orientation(wk2gt_2,141,134). coords(wk2gt_2,98,62,134).  visible(wk2gt_2,134). happensAt(inactive(wk2gt_2),134). orientation(wk2gt_3,154,134). coords(wk2gt_3,182,166,134).  visible(wk2gt_3,134). happensAt(active(wk2gt_3),134).
orientation(wk2gt_2,141,135). coords(wk2gt_2,98,62,135).  visible(wk2gt_2,135). happensAt(inactive(wk2gt_2),135). orientation(wk2gt_3,154,135). coords(wk2gt_3,181,164,135).  visible(wk2gt_3,135). happensAt(active(wk2gt_3),135).
orientation(wk2gt_2,141,136). coords(wk2gt_2,98,62,136).  visible(wk2gt_2,136). happensAt(inactive(wk2gt_2),136). orientation(wk2gt_3,154,136). coords(wk2gt_3,181,162,136).  visible(wk2gt_3,136). happensAt(active(wk2gt_3),136).
orientation(wk2gt_2,141,137). coords(wk2gt_2,98,62,137).  visible(wk2gt_2,137). happensAt(inactive(wk2gt_2),137). orientation(wk2gt_3,154,137). coords(wk2gt_3,181,161,137).  visible(wk2gt_3,137). happensAt(active(wk2gt_3),137).
orientation(wk2gt_2,141,138). coords(wk2gt_2,98,62,138).  visible(wk2gt_2,138). happensAt(inactive(wk2gt_2),138). orientation(wk2gt_3,154,138). coords(wk2gt_3,180,159,138).  visible(wk2gt_3,138). happensAt(active(wk2gt_3),138).
orientation(wk2gt_2,141,139). coords(wk2gt_2,98,62,139).  visible(wk2gt_2,139). happensAt(inactive(wk2gt_2),139). orientation(wk2gt_3,154,139). coords(wk2gt_3,180,159,139).  visible(wk2gt_3,139). happensAt(active(wk2gt_3),139).
orientation(wk2gt_2,141,140). coords(wk2gt_2,98,62,140).  visible(wk2gt_2,140). happensAt(inactive(wk2gt_2),140). orientation(wk2gt_3,154,140). coords(wk2gt_3,179,159,140).  visible(wk2gt_3,140). happensAt(active(wk2gt_3),140).
orientation(wk2gt_2,141,141). coords(wk2gt_2,98,62,141).  visible(wk2gt_2,141). happensAt(inactive(wk2gt_2),141). orientation(wk2gt_3,154,141). coords(wk2gt_3,179,159,141).  visible(wk2gt_3,141). happensAt(active(wk2gt_3),141).
orientation(wk2gt_2,141,142). coords(wk2gt_2,98,62,142).  visible(wk2gt_2,142). happensAt(inactive(wk2gt_2),142). orientation(wk2gt_3,154,142). coords(wk2gt_3,179,158,142).  visible(wk2gt_3,142). happensAt(active(wk2gt_3),142).
#show holdsAt/2.