

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
orientation(wk2gt_1,122,552). coords(wk2gt_1,115,53,552).  visible(wk2gt_1,552). orientation(wk2gt_2,142,552). coords(wk2gt_2,99,63,552).  visible(wk2gt_2,552).
orientation(wk2gt_1,122,553). coords(wk2gt_1,115,53,553).  visible(wk2gt_1,553). orientation(wk2gt_2,142,553). coords(wk2gt_2,99,63,553).  visible(wk2gt_2,553).
orientation(wk2gt_1,122,554). coords(wk2gt_1,115,53,554).  visible(wk2gt_1,554). orientation(wk2gt_2,142,554). coords(wk2gt_2,99,63,554).  visible(wk2gt_2,554).
orientation(wk2gt_1,122,555). coords(wk2gt_1,115,53,555).  visible(wk2gt_1,555). orientation(wk2gt_2,142,555). coords(wk2gt_2,99,63,555).  visible(wk2gt_2,555).
orientation(wk2gt_1,122,556). coords(wk2gt_1,115,53,556).  visible(wk2gt_1,556). orientation(wk2gt_2,142,556). coords(wk2gt_2,99,63,556).  visible(wk2gt_2,556).
orientation(wk2gt_1,122,557). coords(wk2gt_1,115,53,557).  visible(wk2gt_1,557). orientation(wk2gt_2,142,557). coords(wk2gt_2,99,63,557).  visible(wk2gt_2,557).
orientation(wk2gt_1,122,558). coords(wk2gt_1,115,53,558).  visible(wk2gt_1,558). orientation(wk2gt_2,142,558). coords(wk2gt_2,99,63,558).  visible(wk2gt_2,558).
orientation(wk2gt_1,122,559). coords(wk2gt_1,115,53,559).  visible(wk2gt_1,559). orientation(wk2gt_2,142,559). coords(wk2gt_2,99,63,559).  visible(wk2gt_2,559).
orientation(wk2gt_1,122,560). coords(wk2gt_1,115,53,560).  visible(wk2gt_1,560). orientation(wk2gt_2,142,560). coords(wk2gt_2,99,63,560).  visible(wk2gt_2,560).
orientation(wk2gt_1,125,561). coords(wk2gt_1,115,53,561).  visible(wk2gt_1,561). orientation(wk2gt_2,142,561). coords(wk2gt_2,99,63,561).  visible(wk2gt_2,561).
orientation(wk2gt_1,125,562). coords(wk2gt_1,115,53,562).  visible(wk2gt_1,562). orientation(wk2gt_2,142,562). coords(wk2gt_2,99,63,562).  visible(wk2gt_2,562).
orientation(wk2gt_1,125,563). coords(wk2gt_1,115,53,563).  visible(wk2gt_1,563). orientation(wk2gt_2,142,563). coords(wk2gt_2,99,63,563).  visible(wk2gt_2,563).
orientation(wk2gt_1,125,564). coords(wk2gt_1,115,53,564).  visible(wk2gt_1,564). orientation(wk2gt_2,142,564). coords(wk2gt_2,99,63,564).  visible(wk2gt_2,564).
orientation(wk2gt_1,125,565). coords(wk2gt_1,115,53,565).  visible(wk2gt_1,565). orientation(wk2gt_2,142,565). coords(wk2gt_2,99,63,565).  visible(wk2gt_2,565).
orientation(wk2gt_1,125,566). coords(wk2gt_1,115,53,566).  visible(wk2gt_1,566). orientation(wk2gt_2,142,566). coords(wk2gt_2,99,63,566).  visible(wk2gt_2,566).
orientation(wk2gt_1,125,567). coords(wk2gt_1,115,53,567).  visible(wk2gt_1,567). orientation(wk2gt_2,142,567). coords(wk2gt_2,99,63,567).  visible(wk2gt_2,567).
orientation(wk2gt_1,125,568). coords(wk2gt_1,115,53,568).  visible(wk2gt_1,568). orientation(wk2gt_2,142,568). coords(wk2gt_2,99,63,568).  visible(wk2gt_2,568).
orientation(wk2gt_1,125,569). coords(wk2gt_1,115,53,569).  visible(wk2gt_1,569). orientation(wk2gt_2,142,569). coords(wk2gt_2,99,63,569).  visible(wk2gt_2,569).
orientation(wk2gt_1,125,570). coords(wk2gt_1,115,53,570).  visible(wk2gt_1,570). orientation(wk2gt_2,142,570). coords(wk2gt_2,99,63,570).  visible(wk2gt_2,570).
orientation(wk2gt_1,125,571). coords(wk2gt_1,115,53,571).  visible(wk2gt_1,571). orientation(wk2gt_2,142,571). coords(wk2gt_2,99,63,571).  visible(wk2gt_2,571).
orientation(wk2gt_1,125,572). coords(wk2gt_1,115,53,572).  visible(wk2gt_1,572). orientation(wk2gt_2,142,572). coords(wk2gt_2,99,63,572).  visible(wk2gt_2,572).
orientation(wk2gt_1,125,573). coords(wk2gt_1,115,53,573).  visible(wk2gt_1,573). orientation(wk2gt_2,142,573). coords(wk2gt_2,99,63,573).  visible(wk2gt_2,573).
orientation(wk2gt_1,125,574). coords(wk2gt_1,115,53,574).  visible(wk2gt_1,574). orientation(wk2gt_2,142,574). coords(wk2gt_2,99,63,574).  visible(wk2gt_2,574).
orientation(wk2gt_1,125,575). coords(wk2gt_1,115,53,575).  visible(wk2gt_1,575). orientation(wk2gt_2,142,575). coords(wk2gt_2,99,63,575).  visible(wk2gt_2,575).
orientation(wk2gt_1,125,576). coords(wk2gt_1,115,53,576).  visible(wk2gt_1,576). orientation(wk2gt_2,142,576). coords(wk2gt_2,99,63,576).  visible(wk2gt_2,576).
orientation(wk2gt_1,125,577). coords(wk2gt_1,115,53,577).  visible(wk2gt_1,577). orientation(wk2gt_2,142,577). coords(wk2gt_2,99,63,577).  visible(wk2gt_2,577).
happensAt(disappear(wk2gt_1),576). happensAt(disappear(wk2gt_2),576).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=552..576.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=552..576.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),552). :- holdsAt(meeting(wk2gt_2,wk2gt_1),552).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),553). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),553).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),554). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),554).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),555). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),555).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),556). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),556).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),557). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),557).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),558). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),558).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),559). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),559).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),560). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),560).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),561). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),561).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),562). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),562).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),563). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),563).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),564). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),564).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),565). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),565).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),566). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),566).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),567). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),567).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),568). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),568).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),569). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),569).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),570). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),570).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),571). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),571).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),572). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),572).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),573). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),573).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),574). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),574).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),575). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),575).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),576). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),576).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),577). :- holdsAt(meeting(wk2gt_2,wk2gt_1),577).
:- simpleEvent(T,P,walking), T>552 , T<576.
:- simpleEvent(T,P,running), T>552 , T<576.

#show simpleEvent/3.