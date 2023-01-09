

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

	person(mwt2gt_1;mwt2gt_2).
orientation(mwt2gt_1,153,601). coords(mwt2gt_1,28,156,601).  visible(mwt2gt_1,601). orientation(mwt2gt_2,152,601). coords(mwt2gt_2,26,164,601).  visible(mwt2gt_2,601).
orientation(mwt2gt_1,153,602). coords(mwt2gt_1,28,156,602).  visible(mwt2gt_1,602). orientation(mwt2gt_2,152,602). coords(mwt2gt_2,26,164,602).  visible(mwt2gt_2,602).
orientation(mwt2gt_1,153,603). coords(mwt2gt_1,27,156,603).  visible(mwt2gt_1,603). orientation(mwt2gt_2,152,603). coords(mwt2gt_2,26,164,603).  visible(mwt2gt_2,603).
orientation(mwt2gt_1,153,604). coords(mwt2gt_1,29,156,604).  visible(mwt2gt_1,604). orientation(mwt2gt_2,152,604). coords(mwt2gt_2,26,164,604).  visible(mwt2gt_2,604).
orientation(mwt2gt_1,153,605). coords(mwt2gt_1,27,155,605).  visible(mwt2gt_1,605). orientation(mwt2gt_2,152,605). coords(mwt2gt_2,26,164,605).  visible(mwt2gt_2,605).
orientation(mwt2gt_1,153,606). coords(mwt2gt_1,26,155,606).  visible(mwt2gt_1,606). orientation(mwt2gt_2,152,606). coords(mwt2gt_2,26,164,606).  visible(mwt2gt_2,606).
orientation(mwt2gt_1,153,607). coords(mwt2gt_1,26,155,607).  visible(mwt2gt_1,607). orientation(mwt2gt_2,152,607). coords(mwt2gt_2,26,164,607).  visible(mwt2gt_2,607).
orientation(mwt2gt_1,153,608). coords(mwt2gt_1,27,155,608).  visible(mwt2gt_1,608). orientation(mwt2gt_2,152,608). coords(mwt2gt_2,26,164,608).  visible(mwt2gt_2,608).
orientation(mwt2gt_1,153,609). coords(mwt2gt_1,27,155,609).  visible(mwt2gt_1,609). orientation(mwt2gt_2,152,609). coords(mwt2gt_2,26,164,609).  visible(mwt2gt_2,609).
orientation(mwt2gt_1,153,610). coords(mwt2gt_1,26,155,610).  visible(mwt2gt_1,610). orientation(mwt2gt_2,152,610). coords(mwt2gt_2,26,164,610).  visible(mwt2gt_2,610).
orientation(mwt2gt_1,153,611). coords(mwt2gt_1,27,155,611).  visible(mwt2gt_1,611). orientation(mwt2gt_2,152,611). coords(mwt2gt_2,26,164,611).  visible(mwt2gt_2,611).
orientation(mwt2gt_1,153,612). coords(mwt2gt_1,27,155,612).  visible(mwt2gt_1,612). orientation(mwt2gt_2,152,612). coords(mwt2gt_2,26,162,612).  visible(mwt2gt_2,612).
orientation(mwt2gt_1,153,613). coords(mwt2gt_1,27,155,613).  visible(mwt2gt_1,613). orientation(mwt2gt_2,152,613). coords(mwt2gt_2,26,162,613).  visible(mwt2gt_2,613).
orientation(mwt2gt_1,153,614). coords(mwt2gt_1,27,155,614).  visible(mwt2gt_1,614). orientation(mwt2gt_2,152,614). coords(mwt2gt_2,26,160,614).  visible(mwt2gt_2,614).
orientation(mwt2gt_1,153,615). coords(mwt2gt_1,27,153,615).  visible(mwt2gt_1,615). orientation(mwt2gt_2,152,615). coords(mwt2gt_2,26,159,615).  visible(mwt2gt_2,615).
orientation(mwt2gt_1,153,616). coords(mwt2gt_1,27,154,616).  visible(mwt2gt_1,616). orientation(mwt2gt_2,152,616). coords(mwt2gt_2,26,157,616).  visible(mwt2gt_2,616).
orientation(mwt2gt_1,153,617). coords(mwt2gt_1,28,154,617).  visible(mwt2gt_1,617). orientation(mwt2gt_2,152,617). coords(mwt2gt_2,25,156,617).  visible(mwt2gt_2,617).
orientation(mwt2gt_1,153,618). coords(mwt2gt_1,28,152,618).  visible(mwt2gt_1,618). orientation(mwt2gt_2,152,618). coords(mwt2gt_2,25,155,618).  visible(mwt2gt_2,618).
orientation(mwt2gt_1,153,619). coords(mwt2gt_1,27,152,619).  visible(mwt2gt_1,619). orientation(mwt2gt_2,152,619). coords(mwt2gt_2,25,155,619).  visible(mwt2gt_2,619).
orientation(mwt2gt_1,153,620). coords(mwt2gt_1,27,152,620).  visible(mwt2gt_1,620). orientation(mwt2gt_2,152,620). coords(mwt2gt_2,25,155,620).  visible(mwt2gt_2,620).
orientation(mwt2gt_1,153,621). coords(mwt2gt_1,27,153,621).  visible(mwt2gt_1,621). orientation(mwt2gt_2,152,621). coords(mwt2gt_2,25,155,621).  visible(mwt2gt_2,621).
orientation(mwt2gt_1,153,622). coords(mwt2gt_1,27,153,622).  visible(mwt2gt_1,622). orientation(mwt2gt_2,152,622). coords(mwt2gt_2,25,155,622).  visible(mwt2gt_2,622).
orientation(mwt2gt_1,153,623). coords(mwt2gt_1,28,153,623).  visible(mwt2gt_1,623). orientation(mwt2gt_2,152,623). coords(mwt2gt_2,25,155,623).  visible(mwt2gt_2,623).
orientation(mwt2gt_1,153,624). coords(mwt2gt_1,28,153,624).  visible(mwt2gt_1,624). orientation(mwt2gt_2,152,624). coords(mwt2gt_2,25,155,624).  visible(mwt2gt_2,624).
orientation(mwt2gt_1,153,625). coords(mwt2gt_1,28,153,625).  visible(mwt2gt_1,625). orientation(mwt2gt_2,152,625). coords(mwt2gt_2,25,154,625).  visible(mwt2gt_2,625).
orientation(mwt2gt_1,153,626). coords(mwt2gt_1,28,153,626).  visible(mwt2gt_1,626). orientation(mwt2gt_2,152,626). coords(mwt2gt_2,25,154,626).  visible(mwt2gt_2,626).
happensAt(disappear(mwt2gt_1),625). happensAt(disappear(mwt2gt_2),625).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=601..625.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=601..625.
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),601). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),601).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),602). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),602).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),603). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),603).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),604). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),604).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),605). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),605).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),606). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),606).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),607). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),607).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),608). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),608).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),609). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),609).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),610). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),610).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),611). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),611).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),612). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),612).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),613). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),613).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),614). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),614).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),615). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),615).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),616). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),616).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),617). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),617).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),618). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),618).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),619). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),619).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),620). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),620).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),621). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),621).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),622). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),622).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),623). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),623).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),624). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),624).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),625). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),625).
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),626). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),626).
:- simpleEvent(T,P,inactive), T>601 , T<625.
:- simpleEvent(T,P,running), T>601 , T<625.

#show simpleEvent/3.