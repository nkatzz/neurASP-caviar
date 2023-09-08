

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
orientation(wk2gt_1,125,420). coords(wk2gt_1,115,53,420).  visible(wk2gt_1,420). orientation(wk2gt_2,141,420). coords(wk2gt_2,98,63,420).  visible(wk2gt_2,420).
orientation(wk2gt_1,125,421). coords(wk2gt_1,115,53,421).  visible(wk2gt_1,421). orientation(wk2gt_2,141,421). coords(wk2gt_2,98,63,421).  visible(wk2gt_2,421).
orientation(wk2gt_1,125,422). coords(wk2gt_1,115,53,422).  visible(wk2gt_1,422). orientation(wk2gt_2,141,422). coords(wk2gt_2,98,63,422).  visible(wk2gt_2,422).
orientation(wk2gt_1,125,423). coords(wk2gt_1,115,53,423).  visible(wk2gt_1,423). orientation(wk2gt_2,141,423). coords(wk2gt_2,98,63,423).  visible(wk2gt_2,423).
orientation(wk2gt_1,125,424). coords(wk2gt_1,115,53,424).  visible(wk2gt_1,424). orientation(wk2gt_2,141,424). coords(wk2gt_2,98,63,424).  visible(wk2gt_2,424).
orientation(wk2gt_1,125,425). coords(wk2gt_1,115,53,425).  visible(wk2gt_1,425). orientation(wk2gt_2,141,425). coords(wk2gt_2,98,63,425).  visible(wk2gt_2,425).
orientation(wk2gt_1,125,426). coords(wk2gt_1,115,53,426).  visible(wk2gt_1,426). orientation(wk2gt_2,141,426). coords(wk2gt_2,98,63,426).  visible(wk2gt_2,426).
orientation(wk2gt_1,125,427). coords(wk2gt_1,115,53,427).  visible(wk2gt_1,427). orientation(wk2gt_2,141,427). coords(wk2gt_2,98,63,427).  visible(wk2gt_2,427).
orientation(wk2gt_1,125,428). coords(wk2gt_1,115,53,428).  visible(wk2gt_1,428). orientation(wk2gt_2,141,428). coords(wk2gt_2,98,63,428).  visible(wk2gt_2,428).
orientation(wk2gt_1,125,429). coords(wk2gt_1,115,53,429).  visible(wk2gt_1,429). orientation(wk2gt_2,141,429). coords(wk2gt_2,98,63,429).  visible(wk2gt_2,429).
orientation(wk2gt_1,125,430). coords(wk2gt_1,115,53,430).  visible(wk2gt_1,430). orientation(wk2gt_2,141,430). coords(wk2gt_2,98,63,430).  visible(wk2gt_2,430).
orientation(wk2gt_1,125,431). coords(wk2gt_1,115,53,431).  visible(wk2gt_1,431). orientation(wk2gt_2,141,431). coords(wk2gt_2,98,63,431).  visible(wk2gt_2,431).
orientation(wk2gt_1,125,432). coords(wk2gt_1,115,53,432).  visible(wk2gt_1,432). orientation(wk2gt_2,141,432). coords(wk2gt_2,98,63,432).  visible(wk2gt_2,432).
orientation(wk2gt_1,125,433). coords(wk2gt_1,115,53,433).  visible(wk2gt_1,433). orientation(wk2gt_2,141,433). coords(wk2gt_2,98,63,433).  visible(wk2gt_2,433).
orientation(wk2gt_1,125,434). coords(wk2gt_1,115,53,434).  visible(wk2gt_1,434). orientation(wk2gt_2,141,434). coords(wk2gt_2,98,63,434).  visible(wk2gt_2,434).
orientation(wk2gt_1,125,435). coords(wk2gt_1,115,53,435).  visible(wk2gt_1,435). orientation(wk2gt_2,141,435). coords(wk2gt_2,98,63,435).  visible(wk2gt_2,435).
orientation(wk2gt_1,125,436). coords(wk2gt_1,115,53,436).  visible(wk2gt_1,436). orientation(wk2gt_2,141,436). coords(wk2gt_2,98,63,436).  visible(wk2gt_2,436).
orientation(wk2gt_1,125,437). coords(wk2gt_1,115,53,437).  visible(wk2gt_1,437). orientation(wk2gt_2,141,437). coords(wk2gt_2,98,63,437).  visible(wk2gt_2,437).
orientation(wk2gt_1,125,438). coords(wk2gt_1,115,53,438).  visible(wk2gt_1,438). orientation(wk2gt_2,141,438). coords(wk2gt_2,98,63,438).  visible(wk2gt_2,438).
orientation(wk2gt_1,125,439). coords(wk2gt_1,115,53,439).  visible(wk2gt_1,439). orientation(wk2gt_2,141,439). coords(wk2gt_2,98,63,439).  visible(wk2gt_2,439).
orientation(wk2gt_1,125,440). coords(wk2gt_1,115,53,440).  visible(wk2gt_1,440). orientation(wk2gt_2,141,440). coords(wk2gt_2,98,63,440).  visible(wk2gt_2,440).
orientation(wk2gt_1,125,441). coords(wk2gt_1,115,53,441).  visible(wk2gt_1,441). orientation(wk2gt_2,141,441). coords(wk2gt_2,98,63,441).  visible(wk2gt_2,441).
orientation(wk2gt_1,125,442). coords(wk2gt_1,115,53,442).  visible(wk2gt_1,442). orientation(wk2gt_2,141,442). coords(wk2gt_2,98,63,442).  visible(wk2gt_2,442).
orientation(wk2gt_1,125,443). coords(wk2gt_1,115,53,443).  visible(wk2gt_1,443). orientation(wk2gt_2,141,443). coords(wk2gt_2,98,63,443).  visible(wk2gt_2,443).
orientation(wk2gt_1,125,444). coords(wk2gt_1,115,53,444).  visible(wk2gt_1,444). orientation(wk2gt_2,141,444). coords(wk2gt_2,98,63,444).  visible(wk2gt_2,444).
orientation(wk2gt_1,125,445). coords(wk2gt_1,115,53,445).  visible(wk2gt_1,445). orientation(wk2gt_2,141,445). coords(wk2gt_2,98,63,445).  visible(wk2gt_2,445).
happensAt(disappear(wk2gt_1),444). happensAt(disappear(wk2gt_2),444).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=420..444.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=420..444.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),420). :- holdsAt(meeting(wk2gt_2,wk2gt_1),420).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),421). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),421).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),422). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),422).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),423). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),423).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),424). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),424).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),425). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),425).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),426). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),426).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),427). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),427).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),428). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),428).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),429). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),429).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),430). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),430).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),431). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),431).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),432). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),432).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),433). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),433).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),434). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),434).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),435). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),435).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),436). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),436).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),437). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),437).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),438). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),438).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),439). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),439).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),440). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),440).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),441). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),441).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),442). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),442).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),443). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),443).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),444). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),444).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),445). :- holdsAt(meeting(wk2gt_2,wk2gt_1),445).
:- simpleEvent(T,P,walking), T>420 , T<444.
:- simpleEvent(T,P,running), T>420 , T<444.

#show simpleEvent/3.