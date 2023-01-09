

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

	person(wk1gt_4;wk1gt_5).
orientation(wk1gt_4,157,531). coords(wk1gt_4,56,169,531).  visible(wk1gt_4,531). orientation(wk1gt_5,162,531). coords(wk1gt_5,57,150,531).  visible(wk1gt_5,531).
orientation(wk1gt_4,157,532). coords(wk1gt_4,56,169,532).  visible(wk1gt_4,532). orientation(wk1gt_5,162,532). coords(wk1gt_5,57,150,532).  visible(wk1gt_5,532).
orientation(wk1gt_4,157,533). coords(wk1gt_4,56,169,533).  visible(wk1gt_4,533). orientation(wk1gt_5,162,533). coords(wk1gt_5,57,150,533).  visible(wk1gt_5,533).
orientation(wk1gt_4,157,534). coords(wk1gt_4,56,169,534).  visible(wk1gt_4,534). orientation(wk1gt_5,162,534). coords(wk1gt_5,57,150,534).  visible(wk1gt_5,534).
orientation(wk1gt_4,157,535). coords(wk1gt_4,56,169,535).  visible(wk1gt_4,535). orientation(wk1gt_5,162,535). coords(wk1gt_5,57,150,535).  visible(wk1gt_5,535).
orientation(wk1gt_4,157,536). coords(wk1gt_4,56,169,536).  visible(wk1gt_4,536). orientation(wk1gt_5,162,536). coords(wk1gt_5,57,150,536).  visible(wk1gt_5,536).
orientation(wk1gt_4,157,537). coords(wk1gt_4,56,169,537).  visible(wk1gt_4,537). orientation(wk1gt_5,162,537). coords(wk1gt_5,57,150,537).  visible(wk1gt_5,537).
orientation(wk1gt_4,157,538). coords(wk1gt_4,56,169,538).  visible(wk1gt_4,538). orientation(wk1gt_5,162,538). coords(wk1gt_5,57,150,538).  visible(wk1gt_5,538).
orientation(wk1gt_4,157,539). coords(wk1gt_4,56,169,539).  visible(wk1gt_4,539). orientation(wk1gt_5,162,539). coords(wk1gt_5,57,150,539).  visible(wk1gt_5,539).
orientation(wk1gt_4,157,540). coords(wk1gt_4,56,169,540).  visible(wk1gt_4,540). orientation(wk1gt_5,162,540). coords(wk1gt_5,57,150,540).  visible(wk1gt_5,540).
orientation(wk1gt_4,157,541). coords(wk1gt_4,56,169,541).  visible(wk1gt_4,541). orientation(wk1gt_5,162,541). coords(wk1gt_5,57,150,541).  visible(wk1gt_5,541).
orientation(wk1gt_4,157,542). coords(wk1gt_4,56,169,542).  visible(wk1gt_4,542). orientation(wk1gt_5,162,542). coords(wk1gt_5,57,150,542).  visible(wk1gt_5,542).
orientation(wk1gt_4,157,543). coords(wk1gt_4,56,169,543).  visible(wk1gt_4,543). orientation(wk1gt_5,162,543). coords(wk1gt_5,57,150,543).  visible(wk1gt_5,543).
orientation(wk1gt_4,157,544). coords(wk1gt_4,56,169,544).  visible(wk1gt_4,544). orientation(wk1gt_5,162,544). coords(wk1gt_5,57,150,544).  visible(wk1gt_5,544).
orientation(wk1gt_4,157,545). coords(wk1gt_4,56,169,545).  visible(wk1gt_4,545). orientation(wk1gt_5,162,545). coords(wk1gt_5,57,150,545).  visible(wk1gt_5,545).
orientation(wk1gt_4,157,546). coords(wk1gt_4,56,169,546).  visible(wk1gt_4,546). orientation(wk1gt_5,162,546). coords(wk1gt_5,57,150,546).  visible(wk1gt_5,546).
orientation(wk1gt_4,157,547). coords(wk1gt_4,56,169,547).  visible(wk1gt_4,547). orientation(wk1gt_5,162,547). coords(wk1gt_5,57,150,547).  visible(wk1gt_5,547).
orientation(wk1gt_4,157,548). coords(wk1gt_4,56,169,548).  visible(wk1gt_4,548). orientation(wk1gt_5,162,548). coords(wk1gt_5,57,150,548).  visible(wk1gt_5,548).
orientation(wk1gt_4,157,549). coords(wk1gt_4,56,169,549).  visible(wk1gt_4,549). orientation(wk1gt_5,162,549). coords(wk1gt_5,57,150,549).  visible(wk1gt_5,549).
orientation(wk1gt_4,157,550). coords(wk1gt_4,56,169,550).  visible(wk1gt_4,550). orientation(wk1gt_5,162,550). coords(wk1gt_5,57,150,550).  visible(wk1gt_5,550).
orientation(wk1gt_4,157,551). coords(wk1gt_4,56,170,551).  visible(wk1gt_4,551). orientation(wk1gt_5,162,551). coords(wk1gt_5,57,150,551).  visible(wk1gt_5,551).
orientation(wk1gt_4,157,552). coords(wk1gt_4,56,170,552).  visible(wk1gt_4,552). orientation(wk1gt_5,162,552). coords(wk1gt_5,57,150,552).  visible(wk1gt_5,552).
orientation(wk1gt_4,157,553). coords(wk1gt_4,56,170,553).  visible(wk1gt_4,553). orientation(wk1gt_5,162,553). coords(wk1gt_5,57,150,553).  visible(wk1gt_5,553).
orientation(wk1gt_4,157,554). coords(wk1gt_4,56,170,554).  visible(wk1gt_4,554). orientation(wk1gt_5,162,554). coords(wk1gt_5,57,150,554).  visible(wk1gt_5,554).
orientation(wk1gt_4,157,555). coords(wk1gt_4,56,170,555).  visible(wk1gt_4,555). orientation(wk1gt_5,162,555). coords(wk1gt_5,57,150,555).  visible(wk1gt_5,555).
orientation(wk1gt_4,157,556). coords(wk1gt_4,56,170,556).  visible(wk1gt_4,556). orientation(wk1gt_5,162,556). coords(wk1gt_5,57,150,556).  visible(wk1gt_5,556).
happensAt(disappear(wk1gt_4),555). happensAt(disappear(wk1gt_5),555).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=531..555.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=531..555.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),531). :- holdsAt(meeting(wk1gt_5,wk1gt_4),531).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),532). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),532).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),533). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),533).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),534). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),534).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),535). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),535).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),536). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),536).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),537). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),537).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),538). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),538).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),539). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),539).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),540). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),540).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),541). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),541).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),542). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),542).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),543). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),543).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),544). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),544).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),545). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),545).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),546). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),546).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),547). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),547).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),548). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),548).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),549). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),549).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),550). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),550).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),551). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),551).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),552). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),552).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),553). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),553).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),554). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),554).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),555). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),555).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),556). :- holdsAt(meeting(wk1gt_5,wk1gt_4),556).
:- simpleEvent(T,P,walking), T>531 , T<555.
:- simpleEvent(T,P,running), T>531 , T<555.

#show simpleEvent/3.