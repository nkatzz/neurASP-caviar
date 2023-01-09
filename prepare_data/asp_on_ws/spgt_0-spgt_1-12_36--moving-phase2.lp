

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

	person(spgt_0;spgt_1).
orientation(spgt_0,160,12). coords(spgt_0,37,166,12).  visible(spgt_0,12). orientation(spgt_1,160,12). coords(spgt_1,38,157,12).  visible(spgt_1,12).
orientation(spgt_0,160,13). coords(spgt_0,37,166,13).  visible(spgt_0,13). orientation(spgt_1,160,13). coords(spgt_1,38,157,13).  visible(spgt_1,13).
orientation(spgt_0,156,14). coords(spgt_0,37,166,14).  visible(spgt_0,14). orientation(spgt_1,160,14). coords(spgt_1,38,157,14).  visible(spgt_1,14).
orientation(spgt_0,156,15). coords(spgt_0,37,166,15).  visible(spgt_0,15). orientation(spgt_1,160,15). coords(spgt_1,39,157,15).  visible(spgt_1,15).
orientation(spgt_0,156,16). coords(spgt_0,37,167,16).  visible(spgt_0,16). orientation(spgt_1,160,16). coords(spgt_1,39,158,16).  visible(spgt_1,16).
orientation(spgt_0,156,17). coords(spgt_0,38,167,17).  visible(spgt_0,17). orientation(spgt_1,160,17). coords(spgt_1,40,158,17).  visible(spgt_1,17).
orientation(spgt_0,156,18). coords(spgt_0,38,167,18).  visible(spgt_0,18). orientation(spgt_1,160,18). coords(spgt_1,40,158,18).  visible(spgt_1,18).
orientation(spgt_0,156,19). coords(spgt_0,38,167,19).  visible(spgt_0,19). orientation(spgt_1,160,19). coords(spgt_1,40,158,19).  visible(spgt_1,19).
orientation(spgt_0,156,20). coords(spgt_0,38,167,20).  visible(spgt_0,20). orientation(spgt_1,160,20). coords(spgt_1,40,158,20).  visible(spgt_1,20).
orientation(spgt_0,156,21). coords(spgt_0,39,167,21).  visible(spgt_0,21). orientation(spgt_1,160,21). coords(spgt_1,40,158,21).  visible(spgt_1,21).
orientation(spgt_0,156,22). coords(spgt_0,39,167,22).  visible(spgt_0,22). orientation(spgt_1,160,22). coords(spgt_1,40,158,22).  visible(spgt_1,22).
orientation(spgt_0,156,23). coords(spgt_0,41,167,23).  visible(spgt_0,23). orientation(spgt_1,160,23). coords(spgt_1,40,158,23).  visible(spgt_1,23).
orientation(spgt_0,156,24). coords(spgt_0,41,167,24).  visible(spgt_0,24). orientation(spgt_1,160,24). coords(spgt_1,41,158,24).  visible(spgt_1,24).
orientation(spgt_0,156,25). coords(spgt_0,42,167,25).  visible(spgt_0,25). orientation(spgt_1,160,25). coords(spgt_1,41,158,25).  visible(spgt_1,25).
orientation(spgt_0,156,26). coords(spgt_0,42,167,26).  visible(spgt_0,26). orientation(spgt_1,160,26). coords(spgt_1,41,158,26).  visible(spgt_1,26).
orientation(spgt_0,156,27). coords(spgt_0,42,166,27).  visible(spgt_0,27). orientation(spgt_1,160,27). coords(spgt_1,42,158,27).  visible(spgt_1,27).
orientation(spgt_0,156,28). coords(spgt_0,42,166,28).  visible(spgt_0,28). orientation(spgt_1,160,28). coords(spgt_1,42,158,28).  visible(spgt_1,28).
orientation(spgt_0,156,29). coords(spgt_0,42,166,29).  visible(spgt_0,29). orientation(spgt_1,160,29). coords(spgt_1,43,158,29).  visible(spgt_1,29).
orientation(spgt_0,156,30). coords(spgt_0,42,166,30).  visible(spgt_0,30). orientation(spgt_1,160,30). coords(spgt_1,43,158,30).  visible(spgt_1,30).
orientation(spgt_0,156,31). coords(spgt_0,42,166,31).  visible(spgt_0,31). orientation(spgt_1,160,31). coords(spgt_1,43,158,31).  visible(spgt_1,31).
orientation(spgt_0,156,32). coords(spgt_0,42,166,32).  visible(spgt_0,32). orientation(spgt_1,160,32). coords(spgt_1,44,158,32).  visible(spgt_1,32).
orientation(spgt_0,156,33). coords(spgt_0,42,166,33).  visible(spgt_0,33). orientation(spgt_1,160,33). coords(spgt_1,44,158,33).  visible(spgt_1,33).
orientation(spgt_0,156,34). coords(spgt_0,42,166,34).  visible(spgt_0,34). orientation(spgt_1,160,34). coords(spgt_1,45,157,34).  visible(spgt_1,34).
orientation(spgt_0,156,35). coords(spgt_0,44,166,35).  visible(spgt_0,35). orientation(spgt_1,160,35). coords(spgt_1,45,157,35).  visible(spgt_1,35).
orientation(spgt_0,156,36). coords(spgt_0,45,166,36).  visible(spgt_0,36). orientation(spgt_1,160,36). coords(spgt_1,45,157,36).  visible(spgt_1,36).
orientation(spgt_0,156,37). coords(spgt_0,45,166,37).  visible(spgt_0,37). orientation(spgt_1,160,37). coords(spgt_1,45,157,37).  visible(spgt_1,37).
happensAt(disappear(spgt_0),36). happensAt(disappear(spgt_1),36).
1{simpleEvent(T,spgt_0,active); simpleEvent(T,spgt_0,inactive); simpleEvent(T,spgt_0,walking); simpleEvent(T,spgt_0,running)}1 :- T=12..36.
1{simpleEvent(T,spgt_1,active); simpleEvent(T,spgt_1,inactive); simpleEvent(T,spgt_1,walking); simpleEvent(T,spgt_1,running)}1 :- T=12..36.
:- holdsAt(moving(spgt_0,spgt_1),12). :- holdsAt(moving(spgt_1,spgt_0),12).
:- not holdsAt(moving(spgt_0,spgt_1),13). :- not holdsAt(moving(spgt_1,spgt_0),13).
:- not holdsAt(moving(spgt_0,spgt_1),14). :- not holdsAt(moving(spgt_1,spgt_0),14).
:- not holdsAt(moving(spgt_0,spgt_1),15). :- not holdsAt(moving(spgt_1,spgt_0),15).
:- not holdsAt(moving(spgt_0,spgt_1),16). :- not holdsAt(moving(spgt_1,spgt_0),16).
:- not holdsAt(moving(spgt_0,spgt_1),17). :- not holdsAt(moving(spgt_1,spgt_0),17).
:- not holdsAt(moving(spgt_0,spgt_1),18). :- not holdsAt(moving(spgt_1,spgt_0),18).
:- not holdsAt(moving(spgt_0,spgt_1),19). :- not holdsAt(moving(spgt_1,spgt_0),19).
:- not holdsAt(moving(spgt_0,spgt_1),20). :- not holdsAt(moving(spgt_1,spgt_0),20).
:- not holdsAt(moving(spgt_0,spgt_1),21). :- not holdsAt(moving(spgt_1,spgt_0),21).
:- not holdsAt(moving(spgt_0,spgt_1),22). :- not holdsAt(moving(spgt_1,spgt_0),22).
:- not holdsAt(moving(spgt_0,spgt_1),23). :- not holdsAt(moving(spgt_1,spgt_0),23).
:- not holdsAt(moving(spgt_0,spgt_1),24). :- not holdsAt(moving(spgt_1,spgt_0),24).
:- not holdsAt(moving(spgt_0,spgt_1),25). :- not holdsAt(moving(spgt_1,spgt_0),25).
:- not holdsAt(moving(spgt_0,spgt_1),26). :- not holdsAt(moving(spgt_1,spgt_0),26).
:- not holdsAt(moving(spgt_0,spgt_1),27). :- not holdsAt(moving(spgt_1,spgt_0),27).
:- not holdsAt(moving(spgt_0,spgt_1),28). :- not holdsAt(moving(spgt_1,spgt_0),28).
:- not holdsAt(moving(spgt_0,spgt_1),29). :- not holdsAt(moving(spgt_1,spgt_0),29).
:- not holdsAt(moving(spgt_0,spgt_1),30). :- not holdsAt(moving(spgt_1,spgt_0),30).
:- not holdsAt(moving(spgt_0,spgt_1),31). :- not holdsAt(moving(spgt_1,spgt_0),31).
:- not holdsAt(moving(spgt_0,spgt_1),32). :- not holdsAt(moving(spgt_1,spgt_0),32).
:- not holdsAt(moving(spgt_0,spgt_1),33). :- not holdsAt(moving(spgt_1,spgt_0),33).
:- not holdsAt(moving(spgt_0,spgt_1),34). :- not holdsAt(moving(spgt_1,spgt_0),34).
:- not holdsAt(moving(spgt_0,spgt_1),35). :- not holdsAt(moving(spgt_1,spgt_0),35).
:- not holdsAt(moving(spgt_0,spgt_1),36). :- not holdsAt(moving(spgt_1,spgt_0),36).
:- holdsAt(moving(spgt_0,spgt_1),37). :- holdsAt(moving(spgt_1,spgt_0),37).
:- simpleEvent(T,P,inactive), T>12 , T<36.
:- simpleEvent(T,P,running), T>12 , T<36.

#show simpleEvent/3.