
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

	orientation(fcgt_0,152,120). coords(fcgt_0,253,236,120).  visible(fcgt_0,120). happensAt(walking(fcgt_0),120). orientation(fcgt_4,165,120). coords(fcgt_4,101,210,120).  visible(fcgt_4,120). happensAt(inactive(fcgt_4),120).
orientation(fcgt_0,152,121). coords(fcgt_0,255,238,121).  visible(fcgt_0,121). happensAt(walking(fcgt_0),121). orientation(fcgt_4,165,121). coords(fcgt_4,101,210,121).  visible(fcgt_4,121). happensAt(inactive(fcgt_4),121).
orientation(fcgt_0,152,122). coords(fcgt_0,258,239,122).  visible(fcgt_0,122). happensAt(walking(fcgt_0),122). orientation(fcgt_4,165,122). coords(fcgt_4,101,210,122).  visible(fcgt_4,122). happensAt(inactive(fcgt_4),122).
orientation(fcgt_0,152,123). coords(fcgt_0,259,240,123).  visible(fcgt_0,123). happensAt(walking(fcgt_0),123). orientation(fcgt_4,165,123). coords(fcgt_4,101,210,123).  visible(fcgt_4,123). happensAt(inactive(fcgt_4),123).
orientation(fcgt_0,152,124). coords(fcgt_0,261,243,124).  visible(fcgt_0,124). happensAt(walking(fcgt_0),124). orientation(fcgt_4,165,124). coords(fcgt_4,101,210,124).  visible(fcgt_4,124). happensAt(inactive(fcgt_4),124).
orientation(fcgt_0,152,125). coords(fcgt_0,262,245,125).  visible(fcgt_0,125). happensAt(walking(fcgt_0),125). orientation(fcgt_4,165,125). coords(fcgt_4,101,210,125).  visible(fcgt_4,125). happensAt(inactive(fcgt_4),125).
orientation(fcgt_0,152,126). coords(fcgt_0,263,249,126).  visible(fcgt_0,126). happensAt(walking(fcgt_0),126). orientation(fcgt_4,165,126). coords(fcgt_4,101,210,126).  visible(fcgt_4,126). happensAt(inactive(fcgt_4),126).
orientation(fcgt_0,152,127). coords(fcgt_0,264,252,127).  visible(fcgt_0,127). happensAt(walking(fcgt_0),127). orientation(fcgt_4,165,127). coords(fcgt_4,101,210,127).  visible(fcgt_4,127). happensAt(inactive(fcgt_4),127).
orientation(fcgt_0,152,128). coords(fcgt_0,264,255,128).  visible(fcgt_0,128). happensAt(walking(fcgt_0),128). orientation(fcgt_4,165,128). coords(fcgt_4,101,210,128).  visible(fcgt_4,128). happensAt(inactive(fcgt_4),128).
orientation(fcgt_0,152,129). coords(fcgt_0,267,259,129).  visible(fcgt_0,129). happensAt(walking(fcgt_0),129). orientation(fcgt_4,165,129). coords(fcgt_4,101,210,129).  visible(fcgt_4,129). happensAt(inactive(fcgt_4),129).
orientation(fcgt_0,152,130). coords(fcgt_0,269,263,130).  visible(fcgt_0,130). happensAt(walking(fcgt_0),130). orientation(fcgt_4,165,130). coords(fcgt_4,101,210,130).  visible(fcgt_4,130). happensAt(inactive(fcgt_4),130).
orientation(fcgt_0,152,131). coords(fcgt_0,272,266,131).  visible(fcgt_0,131). happensAt(walking(fcgt_0),131). orientation(fcgt_4,165,131). coords(fcgt_4,101,210,131).  visible(fcgt_4,131). happensAt(inactive(fcgt_4),131).
orientation(fcgt_0,152,132). coords(fcgt_0,277,268,132).  visible(fcgt_0,132). happensAt(walking(fcgt_0),132). orientation(fcgt_4,165,132). coords(fcgt_4,101,210,132).  visible(fcgt_4,132). happensAt(inactive(fcgt_4),132).
orientation(fcgt_0,152,133). coords(fcgt_0,280,268,133).  visible(fcgt_0,133). happensAt(walking(fcgt_0),133). orientation(fcgt_4,165,133). coords(fcgt_4,101,210,133).  visible(fcgt_4,133). happensAt(inactive(fcgt_4),133).
orientation(fcgt_0,152,134). coords(fcgt_0,283,268,134).  visible(fcgt_0,134). happensAt(walking(fcgt_0),134). orientation(fcgt_4,170,134). coords(fcgt_4,101,210,134).  visible(fcgt_4,134). happensAt(inactive(fcgt_4),134).
orientation(fcgt_0,152,135). coords(fcgt_0,285,268,135).  visible(fcgt_0,135). happensAt(walking(fcgt_0),135). orientation(fcgt_4,170,135). coords(fcgt_4,101,210,135).  visible(fcgt_4,135). happensAt(inactive(fcgt_4),135).
orientation(fcgt_0,152,136). coords(fcgt_0,287,268,136).  visible(fcgt_0,136). happensAt(walking(fcgt_0),136). orientation(fcgt_4,170,136). coords(fcgt_4,101,210,136).  visible(fcgt_4,136). happensAt(inactive(fcgt_4),136).
orientation(fcgt_0,152,137). coords(fcgt_0,288,268,137).  visible(fcgt_0,137). happensAt(walking(fcgt_0),137). orientation(fcgt_4,170,137). coords(fcgt_4,101,210,137).  visible(fcgt_4,137). happensAt(inactive(fcgt_4),137).
orientation(fcgt_0,152,138). coords(fcgt_0,289,269,138).  visible(fcgt_0,138). happensAt(walking(fcgt_0),138). orientation(fcgt_4,170,138). coords(fcgt_4,101,210,138).  visible(fcgt_4,138). happensAt(inactive(fcgt_4),138).
orientation(fcgt_0,152,139). coords(fcgt_0,290,271,139).  visible(fcgt_0,139). happensAt(walking(fcgt_0),139). orientation(fcgt_4,170,139). coords(fcgt_4,101,210,139).  visible(fcgt_4,139). happensAt(inactive(fcgt_4),139).
orientation(fcgt_0,152,140). coords(fcgt_0,291,273,140).  visible(fcgt_0,140). happensAt(walking(fcgt_0),140). orientation(fcgt_4,170,140). coords(fcgt_4,101,210,140).  visible(fcgt_4,140). happensAt(inactive(fcgt_4),140).
orientation(fcgt_0,152,141). coords(fcgt_0,292,274,141).  visible(fcgt_0,141). happensAt(walking(fcgt_0),141). orientation(fcgt_4,170,141). coords(fcgt_4,101,210,141).  visible(fcgt_4,141). happensAt(inactive(fcgt_4),141).
orientation(fcgt_0,152,142). coords(fcgt_0,293,276,142).  visible(fcgt_0,142). happensAt(walking(fcgt_0),142). orientation(fcgt_4,170,142). coords(fcgt_4,101,210,142).  visible(fcgt_4,142). happensAt(inactive(fcgt_4),142).
orientation(fcgt_0,152,143). coords(fcgt_0,294,278,143).  visible(fcgt_0,143). happensAt(walking(fcgt_0),143). orientation(fcgt_4,170,143). coords(fcgt_4,101,210,143).  visible(fcgt_4,143). happensAt(inactive(fcgt_4),143).
orientation(fcgt_0,152,144). coords(fcgt_0,297,278,144).  visible(fcgt_0,144). happensAt(walking(fcgt_0),144). orientation(fcgt_4,170,144). coords(fcgt_4,101,210,144).  visible(fcgt_4,144). happensAt(inactive(fcgt_4),144).
#show holdsAt/2.