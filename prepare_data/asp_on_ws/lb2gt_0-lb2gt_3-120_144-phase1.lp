
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
orientation(lb2gt_0,141,133). coords(lb2gt_0,192,125,133).  visible(lb2gt_0,133). happensAt(active(lb2gt_0),133). orientation(lb2gt_3,160,133). coords(lb2gt_3,7,59,133).  visible(lb2gt_3,133). happensAt(walking(lb2gt_3),133).
orientation(lb2gt_0,141,134). coords(lb2gt_0,191,125,134).  visible(lb2gt_0,134). happensAt(active(lb2gt_0),134). orientation(lb2gt_3,163,134). coords(lb2gt_3,7,60,134).  visible(lb2gt_3,134). happensAt(walking(lb2gt_3),134).
orientation(lb2gt_0,141,135). coords(lb2gt_0,190,123,135).  visible(lb2gt_0,135). happensAt(active(lb2gt_0),135). orientation(lb2gt_3,162,135). coords(lb2gt_3,7,60,135).  visible(lb2gt_3,135). happensAt(walking(lb2gt_3),135).
orientation(lb2gt_0,141,136). coords(lb2gt_0,189,121,136).  visible(lb2gt_0,136). happensAt(active(lb2gt_0),136). orientation(lb2gt_3,163,136). coords(lb2gt_3,6,60,136).  visible(lb2gt_3,136). happensAt(walking(lb2gt_3),136).
orientation(lb2gt_0,141,137). coords(lb2gt_0,188,120,137).  visible(lb2gt_0,137). happensAt(active(lb2gt_0),137). orientation(lb2gt_3,167,137). coords(lb2gt_3,6,60,137).  visible(lb2gt_3,137). happensAt(walking(lb2gt_3),137).
orientation(lb2gt_0,141,138). coords(lb2gt_0,187,120,138).  visible(lb2gt_0,138). happensAt(active(lb2gt_0),138). orientation(lb2gt_3,159,138). coords(lb2gt_3,6,60,138).  visible(lb2gt_3,138). happensAt(walking(lb2gt_3),138).
orientation(lb2gt_0,141,139). coords(lb2gt_0,187,119,139).  visible(lb2gt_0,139). happensAt(active(lb2gt_0),139). orientation(lb2gt_3,161,139). coords(lb2gt_3,6,60,139).  visible(lb2gt_3,139). happensAt(walking(lb2gt_3),139).
orientation(lb2gt_0,141,140). coords(lb2gt_0,186,119,140).  visible(lb2gt_0,140). happensAt(active(lb2gt_0),140). orientation(lb2gt_3,166,140). coords(lb2gt_3,6,60,140).  visible(lb2gt_3,140). happensAt(walking(lb2gt_3),140).
orientation(lb2gt_0,141,141). coords(lb2gt_0,186,118,141).  visible(lb2gt_0,141). happensAt(active(lb2gt_0),141). orientation(lb2gt_3,160,141). coords(lb2gt_3,6,61,141).  visible(lb2gt_3,141). happensAt(walking(lb2gt_3),141).
orientation(lb2gt_0,141,142). coords(lb2gt_0,185,118,142).  visible(lb2gt_0,142). happensAt(active(lb2gt_0),142). orientation(lb2gt_3,161,142). coords(lb2gt_3,6,61,142).  visible(lb2gt_3,142). happensAt(walking(lb2gt_3),142).
orientation(lb2gt_0,141,143). coords(lb2gt_0,185,117,143).  visible(lb2gt_0,143). happensAt(active(lb2gt_0),143). orientation(lb2gt_3,160,143). coords(lb2gt_3,6,61,143).  visible(lb2gt_3,143). happensAt(walking(lb2gt_3),143).
orientation(lb2gt_0,141,144). coords(lb2gt_0,185,117,144).  visible(lb2gt_0,144). happensAt(active(lb2gt_0),144). orientation(lb2gt_3,159,144). coords(lb2gt_3,6,61,144).  visible(lb2gt_3,144). happensAt(walking(lb2gt_3),144).
#show holdsAt/2.