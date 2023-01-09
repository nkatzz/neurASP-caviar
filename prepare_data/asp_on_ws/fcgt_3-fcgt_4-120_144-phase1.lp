
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

	orientation(fcgt_3,159,120). coords(fcgt_3,103,163,120).  visible(fcgt_3,120). happensAt(inactive(fcgt_3),120). orientation(fcgt_4,165,120). coords(fcgt_4,101,210,120).  visible(fcgt_4,120). happensAt(inactive(fcgt_4),120).
orientation(fcgt_3,159,121). coords(fcgt_3,103,163,121).  visible(fcgt_3,121). happensAt(inactive(fcgt_3),121). orientation(fcgt_4,165,121). coords(fcgt_4,101,210,121).  visible(fcgt_4,121). happensAt(inactive(fcgt_4),121).
orientation(fcgt_3,159,122). coords(fcgt_3,103,163,122).  visible(fcgt_3,122). happensAt(inactive(fcgt_3),122). orientation(fcgt_4,165,122). coords(fcgt_4,101,210,122).  visible(fcgt_4,122). happensAt(inactive(fcgt_4),122).
orientation(fcgt_3,159,123). coords(fcgt_3,103,163,123).  visible(fcgt_3,123). happensAt(inactive(fcgt_3),123). orientation(fcgt_4,165,123). coords(fcgt_4,101,210,123).  visible(fcgt_4,123). happensAt(inactive(fcgt_4),123).
orientation(fcgt_3,159,124). coords(fcgt_3,103,163,124).  visible(fcgt_3,124). happensAt(inactive(fcgt_3),124). orientation(fcgt_4,165,124). coords(fcgt_4,101,210,124).  visible(fcgt_4,124). happensAt(inactive(fcgt_4),124).
orientation(fcgt_3,159,125). coords(fcgt_3,103,163,125).  visible(fcgt_3,125). happensAt(inactive(fcgt_3),125). orientation(fcgt_4,165,125). coords(fcgt_4,101,210,125).  visible(fcgt_4,125). happensAt(inactive(fcgt_4),125).
orientation(fcgt_3,159,126). coords(fcgt_3,103,163,126).  visible(fcgt_3,126). happensAt(inactive(fcgt_3),126). orientation(fcgt_4,165,126). coords(fcgt_4,101,210,126).  visible(fcgt_4,126). happensAt(inactive(fcgt_4),126).
orientation(fcgt_3,159,127). coords(fcgt_3,103,163,127).  visible(fcgt_3,127). happensAt(inactive(fcgt_3),127). orientation(fcgt_4,165,127). coords(fcgt_4,101,210,127).  visible(fcgt_4,127). happensAt(inactive(fcgt_4),127).
orientation(fcgt_3,159,128). coords(fcgt_3,103,163,128).  visible(fcgt_3,128). happensAt(inactive(fcgt_3),128). orientation(fcgt_4,165,128). coords(fcgt_4,101,210,128).  visible(fcgt_4,128). happensAt(inactive(fcgt_4),128).
orientation(fcgt_3,159,129). coords(fcgt_3,103,163,129).  visible(fcgt_3,129). happensAt(inactive(fcgt_3),129). orientation(fcgt_4,165,129). coords(fcgt_4,101,210,129).  visible(fcgt_4,129). happensAt(inactive(fcgt_4),129).
orientation(fcgt_3,159,130). coords(fcgt_3,103,163,130).  visible(fcgt_3,130). happensAt(inactive(fcgt_3),130). orientation(fcgt_4,165,130). coords(fcgt_4,101,210,130).  visible(fcgt_4,130). happensAt(inactive(fcgt_4),130).
orientation(fcgt_3,159,131). coords(fcgt_3,103,163,131).  visible(fcgt_3,131). happensAt(inactive(fcgt_3),131). orientation(fcgt_4,165,131). coords(fcgt_4,101,210,131).  visible(fcgt_4,131). happensAt(inactive(fcgt_4),131).
orientation(fcgt_3,159,132). coords(fcgt_3,103,163,132).  visible(fcgt_3,132). happensAt(inactive(fcgt_3),132). orientation(fcgt_4,165,132). coords(fcgt_4,101,210,132).  visible(fcgt_4,132). happensAt(inactive(fcgt_4),132).
orientation(fcgt_3,159,133). coords(fcgt_3,103,163,133).  visible(fcgt_3,133). happensAt(inactive(fcgt_3),133). orientation(fcgt_4,165,133). coords(fcgt_4,101,210,133).  visible(fcgt_4,133). happensAt(inactive(fcgt_4),133).
orientation(fcgt_3,159,134). coords(fcgt_3,103,163,134).  visible(fcgt_3,134). happensAt(inactive(fcgt_3),134). orientation(fcgt_4,170,134). coords(fcgt_4,101,210,134).  visible(fcgt_4,134). happensAt(inactive(fcgt_4),134).
orientation(fcgt_3,159,135). coords(fcgt_3,103,163,135).  visible(fcgt_3,135). happensAt(inactive(fcgt_3),135). orientation(fcgt_4,170,135). coords(fcgt_4,101,210,135).  visible(fcgt_4,135). happensAt(inactive(fcgt_4),135).
orientation(fcgt_3,159,136). coords(fcgt_3,103,163,136).  visible(fcgt_3,136). happensAt(inactive(fcgt_3),136). orientation(fcgt_4,170,136). coords(fcgt_4,101,210,136).  visible(fcgt_4,136). happensAt(inactive(fcgt_4),136).
orientation(fcgt_3,159,137). coords(fcgt_3,103,163,137).  visible(fcgt_3,137). happensAt(inactive(fcgt_3),137). orientation(fcgt_4,170,137). coords(fcgt_4,101,210,137).  visible(fcgt_4,137). happensAt(inactive(fcgt_4),137).
orientation(fcgt_3,159,138). coords(fcgt_3,103,163,138).  visible(fcgt_3,138). happensAt(inactive(fcgt_3),138). orientation(fcgt_4,170,138). coords(fcgt_4,101,210,138).  visible(fcgt_4,138). happensAt(inactive(fcgt_4),138).
orientation(fcgt_3,159,139). coords(fcgt_3,103,163,139).  visible(fcgt_3,139). happensAt(inactive(fcgt_3),139). orientation(fcgt_4,170,139). coords(fcgt_4,101,210,139).  visible(fcgt_4,139). happensAt(inactive(fcgt_4),139).
orientation(fcgt_3,159,140). coords(fcgt_3,103,163,140).  visible(fcgt_3,140). happensAt(inactive(fcgt_3),140). orientation(fcgt_4,170,140). coords(fcgt_4,101,210,140).  visible(fcgt_4,140). happensAt(inactive(fcgt_4),140).
orientation(fcgt_3,159,141). coords(fcgt_3,103,163,141).  visible(fcgt_3,141). happensAt(inactive(fcgt_3),141). orientation(fcgt_4,170,141). coords(fcgt_4,101,210,141).  visible(fcgt_4,141). happensAt(inactive(fcgt_4),141).
orientation(fcgt_3,159,142). coords(fcgt_3,103,163,142).  visible(fcgt_3,142). happensAt(inactive(fcgt_3),142). orientation(fcgt_4,170,142). coords(fcgt_4,101,210,142).  visible(fcgt_4,142). happensAt(inactive(fcgt_4),142).
orientation(fcgt_3,159,143). coords(fcgt_3,103,163,143).  visible(fcgt_3,143). happensAt(inactive(fcgt_3),143). orientation(fcgt_4,170,143). coords(fcgt_4,101,210,143).  visible(fcgt_4,143). happensAt(inactive(fcgt_4),143).
orientation(fcgt_3,159,144). coords(fcgt_3,103,163,144).  visible(fcgt_3,144). happensAt(inactive(fcgt_3),144). orientation(fcgt_4,170,144). coords(fcgt_4,101,210,144).  visible(fcgt_4,144). happensAt(inactive(fcgt_4),144).
#show holdsAt/2.