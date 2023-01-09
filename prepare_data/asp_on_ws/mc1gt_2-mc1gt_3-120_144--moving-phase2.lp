

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

	person(mc1gt_2;mc1gt_3).
orientation(mc1gt_2,144,120). coords(mc1gt_2,77,79,120).  visible(mc1gt_2,120). orientation(mc1gt_3,146,120). coords(mc1gt_3,80,71,120).  visible(mc1gt_3,120).
orientation(mc1gt_2,144,121). coords(mc1gt_2,77,80,121).  visible(mc1gt_2,121). orientation(mc1gt_3,146,121). coords(mc1gt_3,81,71,121).  visible(mc1gt_3,121).
orientation(mc1gt_2,144,122). coords(mc1gt_2,77,80,122).  visible(mc1gt_2,122). orientation(mc1gt_3,146,122). coords(mc1gt_3,83,73,122).  visible(mc1gt_3,122).
orientation(mc1gt_2,144,123). coords(mc1gt_2,77,81,123).  visible(mc1gt_2,123). orientation(mc1gt_3,146,123). coords(mc1gt_3,83,73,123).  visible(mc1gt_3,123).
orientation(mc1gt_2,144,124). coords(mc1gt_2,77,83,124).  visible(mc1gt_2,124). orientation(mc1gt_3,146,124). coords(mc1gt_3,83,73,124).  visible(mc1gt_3,124).
orientation(mc1gt_2,144,125). coords(mc1gt_2,76,85,125).  visible(mc1gt_2,125). orientation(mc1gt_3,146,125). coords(mc1gt_3,83,73,125).  visible(mc1gt_3,125).
orientation(mc1gt_2,144,126). coords(mc1gt_2,76,86,126).  visible(mc1gt_2,126). orientation(mc1gt_3,146,126). coords(mc1gt_3,84,74,126).  visible(mc1gt_3,126).
orientation(mc1gt_2,144,127). coords(mc1gt_2,76,86,127).  visible(mc1gt_2,127). orientation(mc1gt_3,146,127). coords(mc1gt_3,84,74,127).  visible(mc1gt_3,127).
orientation(mc1gt_2,144,128). coords(mc1gt_2,76,87,128).  visible(mc1gt_2,128). orientation(mc1gt_3,146,128). coords(mc1gt_3,84,74,128).  visible(mc1gt_3,128).
orientation(mc1gt_2,144,129). coords(mc1gt_2,76,87,129).  visible(mc1gt_2,129). orientation(mc1gt_3,146,129). coords(mc1gt_3,84,75,129).  visible(mc1gt_3,129).
orientation(mc1gt_2,144,130). coords(mc1gt_2,76,88,130).  visible(mc1gt_2,130). orientation(mc1gt_3,146,130). coords(mc1gt_3,85,76,130).  visible(mc1gt_3,130).
orientation(mc1gt_2,144,131). coords(mc1gt_2,76,88,131).  visible(mc1gt_2,131). orientation(mc1gt_3,146,131). coords(mc1gt_3,85,77,131).  visible(mc1gt_3,131).
orientation(mc1gt_2,144,132). coords(mc1gt_2,76,88,132).  visible(mc1gt_2,132). orientation(mc1gt_3,146,132). coords(mc1gt_3,86,77,132).  visible(mc1gt_3,132).
orientation(mc1gt_2,144,133). coords(mc1gt_2,76,89,133).  visible(mc1gt_2,133). orientation(mc1gt_3,146,133). coords(mc1gt_3,87,79,133).  visible(mc1gt_3,133).
orientation(mc1gt_2,144,134). coords(mc1gt_2,76,89,134).  visible(mc1gt_2,134). orientation(mc1gt_3,146,134). coords(mc1gt_3,89,79,134).  visible(mc1gt_3,134).
orientation(mc1gt_2,144,135). coords(mc1gt_2,76,90,135).  visible(mc1gt_2,135). orientation(mc1gt_3,146,135). coords(mc1gt_3,89,78,135).  visible(mc1gt_3,135).
orientation(mc1gt_2,144,136). coords(mc1gt_2,76,90,136).  visible(mc1gt_2,136). orientation(mc1gt_3,146,136). coords(mc1gt_3,89,78,136).  visible(mc1gt_3,136).
orientation(mc1gt_2,144,137). coords(mc1gt_2,77,90,137).  visible(mc1gt_2,137). orientation(mc1gt_3,146,137). coords(mc1gt_3,90,78,137).  visible(mc1gt_3,137).
orientation(mc1gt_2,144,138). coords(mc1gt_2,77,91,138).  visible(mc1gt_2,138). orientation(mc1gt_3,146,138). coords(mc1gt_3,91,79,138).  visible(mc1gt_3,138).
orientation(mc1gt_2,144,139). coords(mc1gt_2,77,92,139).  visible(mc1gt_2,139). orientation(mc1gt_3,146,139). coords(mc1gt_3,91,79,139).  visible(mc1gt_3,139).
orientation(mc1gt_2,144,140). coords(mc1gt_2,76,92,140).  visible(mc1gt_2,140). orientation(mc1gt_3,146,140). coords(mc1gt_3,91,79,140).  visible(mc1gt_3,140).
orientation(mc1gt_2,144,141). coords(mc1gt_2,75,93,141).  visible(mc1gt_2,141). orientation(mc1gt_3,146,141). coords(mc1gt_3,91,79,141).  visible(mc1gt_3,141).
orientation(mc1gt_2,144,142). coords(mc1gt_2,75,94,142).  visible(mc1gt_2,142). orientation(mc1gt_3,146,142). coords(mc1gt_3,91,79,142).  visible(mc1gt_3,142).
orientation(mc1gt_2,144,143). coords(mc1gt_2,73,95,143).  visible(mc1gt_2,143). orientation(mc1gt_3,146,143). coords(mc1gt_3,91,79,143).  visible(mc1gt_3,143).
orientation(mc1gt_2,144,144). coords(mc1gt_2,73,95,144).  visible(mc1gt_2,144). orientation(mc1gt_3,146,144). coords(mc1gt_3,91,79,144).  visible(mc1gt_3,144).
orientation(mc1gt_2,144,145). coords(mc1gt_2,73,95,145).  visible(mc1gt_2,145). orientation(mc1gt_3,146,145). coords(mc1gt_3,91,79,145).  visible(mc1gt_3,145).
happensAt(disappear(mc1gt_2),144). happensAt(disappear(mc1gt_3),144).
1{simpleEvent(T,mc1gt_2,active); simpleEvent(T,mc1gt_2,inactive); simpleEvent(T,mc1gt_2,walking); simpleEvent(T,mc1gt_2,running)}1 :- T=120..144.
1{simpleEvent(T,mc1gt_3,active); simpleEvent(T,mc1gt_3,inactive); simpleEvent(T,mc1gt_3,walking); simpleEvent(T,mc1gt_3,running)}1 :- T=120..144.
:- holdsAt(moving(mc1gt_2,mc1gt_3),120). :- holdsAt(moving(mc1gt_3,mc1gt_2),120).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),121). :- not holdsAt(moving(mc1gt_3,mc1gt_2),121).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),122). :- not holdsAt(moving(mc1gt_3,mc1gt_2),122).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),123). :- not holdsAt(moving(mc1gt_3,mc1gt_2),123).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),124). :- not holdsAt(moving(mc1gt_3,mc1gt_2),124).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),125). :- not holdsAt(moving(mc1gt_3,mc1gt_2),125).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),126). :- not holdsAt(moving(mc1gt_3,mc1gt_2),126).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),127). :- not holdsAt(moving(mc1gt_3,mc1gt_2),127).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),128). :- not holdsAt(moving(mc1gt_3,mc1gt_2),128).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),129). :- not holdsAt(moving(mc1gt_3,mc1gt_2),129).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),130). :- not holdsAt(moving(mc1gt_3,mc1gt_2),130).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),131). :- not holdsAt(moving(mc1gt_3,mc1gt_2),131).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),132). :- not holdsAt(moving(mc1gt_3,mc1gt_2),132).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),133). :- not holdsAt(moving(mc1gt_3,mc1gt_2),133).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),134). :- not holdsAt(moving(mc1gt_3,mc1gt_2),134).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),135). :- not holdsAt(moving(mc1gt_3,mc1gt_2),135).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),136). :- not holdsAt(moving(mc1gt_3,mc1gt_2),136).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),137). :- not holdsAt(moving(mc1gt_3,mc1gt_2),137).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),138). :- not holdsAt(moving(mc1gt_3,mc1gt_2),138).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),139). :- not holdsAt(moving(mc1gt_3,mc1gt_2),139).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),140). :- not holdsAt(moving(mc1gt_3,mc1gt_2),140).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),141). :- not holdsAt(moving(mc1gt_3,mc1gt_2),141).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),142). :- not holdsAt(moving(mc1gt_3,mc1gt_2),142).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),143). :- not holdsAt(moving(mc1gt_3,mc1gt_2),143).
:- not holdsAt(moving(mc1gt_2,mc1gt_3),144). :- not holdsAt(moving(mc1gt_3,mc1gt_2),144).
:- holdsAt(moving(mc1gt_2,mc1gt_3),145). :- holdsAt(moving(mc1gt_3,mc1gt_2),145).
:- simpleEvent(T,P,inactive), T>120 , T<144.
:- simpleEvent(T,P,running), T>120 , T<144.

#show simpleEvent/3.