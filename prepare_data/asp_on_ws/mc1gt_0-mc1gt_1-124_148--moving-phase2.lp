

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

	person(mc1gt_0;mc1gt_1).
orientation(mc1gt_0,147,124). coords(mc1gt_0,87,102,124).  visible(mc1gt_0,124). orientation(mc1gt_1,145,124). coords(mc1gt_1,107,82,124).  visible(mc1gt_1,124).
orientation(mc1gt_0,147,125). coords(mc1gt_0,87,102,125).  visible(mc1gt_0,125). orientation(mc1gt_1,145,125). coords(mc1gt_1,108,82,125).  visible(mc1gt_1,125).
orientation(mc1gt_0,147,126). coords(mc1gt_0,88,103,126).  visible(mc1gt_0,126). orientation(mc1gt_1,145,126). coords(mc1gt_1,109,82,126).  visible(mc1gt_1,126).
orientation(mc1gt_0,147,127). coords(mc1gt_0,89,103,127).  visible(mc1gt_0,127). orientation(mc1gt_1,145,127). coords(mc1gt_1,110,82,127).  visible(mc1gt_1,127).
orientation(mc1gt_0,147,128). coords(mc1gt_0,89,103,128).  visible(mc1gt_0,128). orientation(mc1gt_1,145,128). coords(mc1gt_1,110,82,128).  visible(mc1gt_1,128).
orientation(mc1gt_0,147,129). coords(mc1gt_0,89,103,129).  visible(mc1gt_0,129). orientation(mc1gt_1,145,129). coords(mc1gt_1,110,82,129).  visible(mc1gt_1,129).
orientation(mc1gt_0,147,130). coords(mc1gt_0,89,103,130).  visible(mc1gt_0,130). orientation(mc1gt_1,145,130). coords(mc1gt_1,111,82,130).  visible(mc1gt_1,130).
orientation(mc1gt_0,147,131). coords(mc1gt_0,89,104,131).  visible(mc1gt_0,131). orientation(mc1gt_1,145,131). coords(mc1gt_1,111,81,131).  visible(mc1gt_1,131).
orientation(mc1gt_0,147,132). coords(mc1gt_0,89,104,132).  visible(mc1gt_0,132). orientation(mc1gt_1,145,132). coords(mc1gt_1,112,81,132).  visible(mc1gt_1,132).
orientation(mc1gt_0,147,133). coords(mc1gt_0,90,105,133).  visible(mc1gt_0,133). orientation(mc1gt_1,145,133). coords(mc1gt_1,112,81,133).  visible(mc1gt_1,133).
orientation(mc1gt_0,147,134). coords(mc1gt_0,90,105,134).  visible(mc1gt_0,134). orientation(mc1gt_1,145,134). coords(mc1gt_1,112,81,134).  visible(mc1gt_1,134).
orientation(mc1gt_0,147,135). coords(mc1gt_0,91,106,135).  visible(mc1gt_0,135). orientation(mc1gt_1,145,135). coords(mc1gt_1,113,81,135).  visible(mc1gt_1,135).
orientation(mc1gt_0,147,136). coords(mc1gt_0,92,106,136).  visible(mc1gt_0,136). orientation(mc1gt_1,145,136). coords(mc1gt_1,113,81,136).  visible(mc1gt_1,136).
orientation(mc1gt_0,147,137). coords(mc1gt_0,93,106,137).  visible(mc1gt_0,137). orientation(mc1gt_1,145,137). coords(mc1gt_1,113,81,137).  visible(mc1gt_1,137).
orientation(mc1gt_0,147,138). coords(mc1gt_0,94,106,138).  visible(mc1gt_0,138). orientation(mc1gt_1,145,138). coords(mc1gt_1,114,82,138).  visible(mc1gt_1,138).
orientation(mc1gt_0,147,139). coords(mc1gt_0,95,106,139).  visible(mc1gt_0,139). orientation(mc1gt_1,145,139). coords(mc1gt_1,116,82,139).  visible(mc1gt_1,139).
orientation(mc1gt_0,147,140). coords(mc1gt_0,95,106,140).  visible(mc1gt_0,140). orientation(mc1gt_1,145,140). coords(mc1gt_1,116,82,140).  visible(mc1gt_1,140).
orientation(mc1gt_0,147,141). coords(mc1gt_0,95,106,141).  visible(mc1gt_0,141). orientation(mc1gt_1,145,141). coords(mc1gt_1,117,82,141).  visible(mc1gt_1,141).
orientation(mc1gt_0,147,142). coords(mc1gt_0,95,106,142).  visible(mc1gt_0,142). orientation(mc1gt_1,145,142). coords(mc1gt_1,118,83,142).  visible(mc1gt_1,142).
orientation(mc1gt_0,147,143). coords(mc1gt_0,95,106,143).  visible(mc1gt_0,143). orientation(mc1gt_1,145,143). coords(mc1gt_1,118,83,143).  visible(mc1gt_1,143).
orientation(mc1gt_0,147,144). coords(mc1gt_0,95,106,144).  visible(mc1gt_0,144). orientation(mc1gt_1,145,144). coords(mc1gt_1,119,83,144).  visible(mc1gt_1,144).
orientation(mc1gt_0,147,145). coords(mc1gt_0,95,107,145).  visible(mc1gt_0,145). orientation(mc1gt_1,145,145). coords(mc1gt_1,119,83,145).  visible(mc1gt_1,145).
orientation(mc1gt_0,147,146). coords(mc1gt_0,96,107,146).  visible(mc1gt_0,146). orientation(mc1gt_1,145,146). coords(mc1gt_1,119,83,146).  visible(mc1gt_1,146).
orientation(mc1gt_0,147,147). coords(mc1gt_0,98,106,147).  visible(mc1gt_0,147). orientation(mc1gt_1,145,147). coords(mc1gt_1,120,83,147).  visible(mc1gt_1,147).
orientation(mc1gt_0,147,148). coords(mc1gt_0,100,107,148).  visible(mc1gt_0,148). orientation(mc1gt_1,145,148). coords(mc1gt_1,120,83,148).  visible(mc1gt_1,148).
orientation(mc1gt_0,147,149). coords(mc1gt_0,100,107,149).  visible(mc1gt_0,149). orientation(mc1gt_1,145,149). coords(mc1gt_1,120,83,149).  visible(mc1gt_1,149).
happensAt(disappear(mc1gt_0),148). happensAt(disappear(mc1gt_1),148).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=124..148.
1{simpleEvent(T,mc1gt_1,active); simpleEvent(T,mc1gt_1,inactive); simpleEvent(T,mc1gt_1,walking); simpleEvent(T,mc1gt_1,running)}1 :- T=124..148.
:- holdsAt(moving(mc1gt_0,mc1gt_1),124). :- holdsAt(moving(mc1gt_1,mc1gt_0),124).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),125). :- not holdsAt(moving(mc1gt_1,mc1gt_0),125).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),126). :- not holdsAt(moving(mc1gt_1,mc1gt_0),126).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),127). :- not holdsAt(moving(mc1gt_1,mc1gt_0),127).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),128). :- not holdsAt(moving(mc1gt_1,mc1gt_0),128).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),129). :- not holdsAt(moving(mc1gt_1,mc1gt_0),129).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),130). :- not holdsAt(moving(mc1gt_1,mc1gt_0),130).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),131). :- not holdsAt(moving(mc1gt_1,mc1gt_0),131).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),132). :- not holdsAt(moving(mc1gt_1,mc1gt_0),132).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),133). :- not holdsAt(moving(mc1gt_1,mc1gt_0),133).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),134). :- not holdsAt(moving(mc1gt_1,mc1gt_0),134).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),135). :- not holdsAt(moving(mc1gt_1,mc1gt_0),135).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),136). :- not holdsAt(moving(mc1gt_1,mc1gt_0),136).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),137). :- not holdsAt(moving(mc1gt_1,mc1gt_0),137).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),138). :- not holdsAt(moving(mc1gt_1,mc1gt_0),138).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),139). :- not holdsAt(moving(mc1gt_1,mc1gt_0),139).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),140). :- not holdsAt(moving(mc1gt_1,mc1gt_0),140).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),141). :- not holdsAt(moving(mc1gt_1,mc1gt_0),141).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),142). :- not holdsAt(moving(mc1gt_1,mc1gt_0),142).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),143). :- not holdsAt(moving(mc1gt_1,mc1gt_0),143).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),144). :- not holdsAt(moving(mc1gt_1,mc1gt_0),144).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),145). :- not holdsAt(moving(mc1gt_1,mc1gt_0),145).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),146). :- not holdsAt(moving(mc1gt_1,mc1gt_0),146).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),147). :- not holdsAt(moving(mc1gt_1,mc1gt_0),147).
:- not holdsAt(moving(mc1gt_0,mc1gt_1),148). :- not holdsAt(moving(mc1gt_1,mc1gt_0),148).
:- holdsAt(moving(mc1gt_0,mc1gt_1),149). :- holdsAt(moving(mc1gt_1,mc1gt_0),149).
:- simpleEvent(T,P,inactive), T>124 , T<148.
:- simpleEvent(T,P,running), T>124 , T<148.

#show simpleEvent/3.