

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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	person(wk2gt_1;wk2gt_2).
orientation(wk2gt_1,132,120). coords(wk2gt_1,114,52,120).  visible(wk2gt_1,120). orientation(wk2gt_2,141,120). coords(wk2gt_2,98,62,120).  visible(wk2gt_2,120).
orientation(wk2gt_1,132,121). coords(wk2gt_1,114,52,121).  visible(wk2gt_1,121). orientation(wk2gt_2,141,121). coords(wk2gt_2,98,62,121).  visible(wk2gt_2,121).
orientation(wk2gt_1,132,122). coords(wk2gt_1,114,52,122).  visible(wk2gt_1,122). orientation(wk2gt_2,141,122). coords(wk2gt_2,98,62,122).  visible(wk2gt_2,122).
orientation(wk2gt_1,132,123). coords(wk2gt_1,114,52,123).  visible(wk2gt_1,123). orientation(wk2gt_2,141,123). coords(wk2gt_2,98,62,123).  visible(wk2gt_2,123).
orientation(wk2gt_1,132,124). coords(wk2gt_1,114,52,124).  visible(wk2gt_1,124). orientation(wk2gt_2,141,124). coords(wk2gt_2,98,62,124).  visible(wk2gt_2,124).
orientation(wk2gt_1,132,125). coords(wk2gt_1,114,52,125).  visible(wk2gt_1,125). orientation(wk2gt_2,141,125). coords(wk2gt_2,98,62,125).  visible(wk2gt_2,125).
orientation(wk2gt_1,132,126). coords(wk2gt_1,114,52,126).  visible(wk2gt_1,126). orientation(wk2gt_2,141,126). coords(wk2gt_2,98,62,126).  visible(wk2gt_2,126).
orientation(wk2gt_1,132,127). coords(wk2gt_1,114,52,127).  visible(wk2gt_1,127). orientation(wk2gt_2,141,127). coords(wk2gt_2,98,62,127).  visible(wk2gt_2,127).
orientation(wk2gt_1,132,128). coords(wk2gt_1,114,52,128).  visible(wk2gt_1,128). orientation(wk2gt_2,141,128). coords(wk2gt_2,98,62,128).  visible(wk2gt_2,128).
orientation(wk2gt_1,132,129). coords(wk2gt_1,114,52,129).  visible(wk2gt_1,129). orientation(wk2gt_2,141,129). coords(wk2gt_2,98,62,129).  visible(wk2gt_2,129).
orientation(wk2gt_1,132,130). coords(wk2gt_1,114,52,130).  visible(wk2gt_1,130). orientation(wk2gt_2,141,130). coords(wk2gt_2,98,62,130).  visible(wk2gt_2,130).
orientation(wk2gt_1,132,131). coords(wk2gt_1,114,52,131).  visible(wk2gt_1,131). orientation(wk2gt_2,141,131). coords(wk2gt_2,98,62,131).  visible(wk2gt_2,131).
orientation(wk2gt_1,132,132). coords(wk2gt_1,114,52,132).  visible(wk2gt_1,132). orientation(wk2gt_2,141,132). coords(wk2gt_2,98,62,132).  visible(wk2gt_2,132).
orientation(wk2gt_1,132,133). coords(wk2gt_1,114,52,133).  visible(wk2gt_1,133). orientation(wk2gt_2,141,133). coords(wk2gt_2,98,62,133).  visible(wk2gt_2,133).
orientation(wk2gt_1,132,134). coords(wk2gt_1,114,52,134).  visible(wk2gt_1,134). orientation(wk2gt_2,141,134). coords(wk2gt_2,98,62,134).  visible(wk2gt_2,134).
orientation(wk2gt_1,132,135). coords(wk2gt_1,114,52,135).  visible(wk2gt_1,135). orientation(wk2gt_2,141,135). coords(wk2gt_2,98,62,135).  visible(wk2gt_2,135).
orientation(wk2gt_1,132,136). coords(wk2gt_1,114,52,136).  visible(wk2gt_1,136). orientation(wk2gt_2,141,136). coords(wk2gt_2,98,62,136).  visible(wk2gt_2,136).
orientation(wk2gt_1,132,137). coords(wk2gt_1,114,52,137).  visible(wk2gt_1,137). orientation(wk2gt_2,141,137). coords(wk2gt_2,98,62,137).  visible(wk2gt_2,137).
orientation(wk2gt_1,132,138). coords(wk2gt_1,114,52,138).  visible(wk2gt_1,138). orientation(wk2gt_2,141,138). coords(wk2gt_2,98,62,138).  visible(wk2gt_2,138).
orientation(wk2gt_1,132,139). coords(wk2gt_1,114,52,139).  visible(wk2gt_1,139). orientation(wk2gt_2,141,139). coords(wk2gt_2,98,62,139).  visible(wk2gt_2,139).
orientation(wk2gt_1,132,140). coords(wk2gt_1,114,52,140).  visible(wk2gt_1,140). orientation(wk2gt_2,141,140). coords(wk2gt_2,98,62,140).  visible(wk2gt_2,140).
orientation(wk2gt_1,132,141). coords(wk2gt_1,114,52,141).  visible(wk2gt_1,141). orientation(wk2gt_2,141,141). coords(wk2gt_2,98,62,141).  visible(wk2gt_2,141).
orientation(wk2gt_1,132,142). coords(wk2gt_1,114,53,142).  visible(wk2gt_1,142). orientation(wk2gt_2,141,142). coords(wk2gt_2,98,62,142).  visible(wk2gt_2,142).
orientation(wk2gt_1,132,143). coords(wk2gt_1,114,53,143).  visible(wk2gt_1,143). orientation(wk2gt_2,141,143). coords(wk2gt_2,98,62,143).  visible(wk2gt_2,143).
orientation(wk2gt_1,132,144). coords(wk2gt_1,114,53,144).  visible(wk2gt_1,144). orientation(wk2gt_2,141,144). coords(wk2gt_2,98,62,144).  visible(wk2gt_2,144).
orientation(wk2gt_1,132,145). coords(wk2gt_1,114,53,145).  visible(wk2gt_1,145). orientation(wk2gt_2,141,145). coords(wk2gt_2,98,62,145).  visible(wk2gt_2,145).
happensAt(disappear(wk2gt_1),144). happensAt(disappear(wk2gt_2),144).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=120..144.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=120..144.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),120). :- holdsAt(meeting(wk2gt_2,wk2gt_1),120).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),121). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),121).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),122). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),122).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),123). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),123).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),124). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),124).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),125). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),125).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),126). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),126).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),127). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),127).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),128). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),128).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),129). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),129).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),130). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),130).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),131). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),131).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),132). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),132).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),133). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),133).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),134). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),134).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),135). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),135).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),136). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),136).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),137). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),137).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),138). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),138).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),139). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),139).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),140). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),140).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),141). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),141).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),142). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),142).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),143). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),143).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),144). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),144).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),145). :- holdsAt(meeting(wk2gt_2,wk2gt_1),145).
:- simpleEvent(T,P,walking), T>120 , T<144.
:- simpleEvent(T,P,running), T>120 , T<144.

#show simpleEvent/3.