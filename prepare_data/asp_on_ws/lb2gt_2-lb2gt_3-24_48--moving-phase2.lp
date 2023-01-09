

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

	person(lb2gt_2;lb2gt_3).
orientation(lb2gt_2,147,24). coords(lb2gt_2,21,50,24).  visible(lb2gt_2,24). orientation(lb2gt_3,146,24). coords(lb2gt_3,23,34,24).  visible(lb2gt_3,24).
orientation(lb2gt_2,147,25). coords(lb2gt_2,20,51,25).  visible(lb2gt_2,25). orientation(lb2gt_3,146,25). coords(lb2gt_3,22,35,25).  visible(lb2gt_3,25).
orientation(lb2gt_2,147,26). coords(lb2gt_2,21,52,26).  visible(lb2gt_2,26). orientation(lb2gt_3,146,26). coords(lb2gt_3,22,35,26).  visible(lb2gt_3,26).
orientation(lb2gt_2,147,27). coords(lb2gt_2,21,52,27).  visible(lb2gt_2,27). orientation(lb2gt_3,146,27). coords(lb2gt_3,22,35,27).  visible(lb2gt_3,27).
orientation(lb2gt_2,147,28). coords(lb2gt_2,21,52,28).  visible(lb2gt_2,28). orientation(lb2gt_3,146,28). coords(lb2gt_3,22,37,28).  visible(lb2gt_3,28).
orientation(lb2gt_2,147,29). coords(lb2gt_2,20,52,29).  visible(lb2gt_2,29). orientation(lb2gt_3,146,29). coords(lb2gt_3,23,37,29).  visible(lb2gt_3,29).
orientation(lb2gt_2,147,30). coords(lb2gt_2,20,52,30).  visible(lb2gt_2,30). orientation(lb2gt_3,146,30). coords(lb2gt_3,24,38,30).  visible(lb2gt_3,30).
orientation(lb2gt_2,147,31). coords(lb2gt_2,20,53,31).  visible(lb2gt_2,31). orientation(lb2gt_3,146,31). coords(lb2gt_3,24,39,31).  visible(lb2gt_3,31).
orientation(lb2gt_2,147,32). coords(lb2gt_2,20,53,32).  visible(lb2gt_2,32). orientation(lb2gt_3,146,32). coords(lb2gt_3,24,39,32).  visible(lb2gt_3,32).
orientation(lb2gt_2,147,33). coords(lb2gt_2,20,53,33).  visible(lb2gt_2,33). orientation(lb2gt_3,146,33). coords(lb2gt_3,24,39,33).  visible(lb2gt_3,33).
orientation(lb2gt_2,147,34). coords(lb2gt_2,20,53,34).  visible(lb2gt_2,34). orientation(lb2gt_3,146,34). coords(lb2gt_3,24,39,34).  visible(lb2gt_3,34).
orientation(lb2gt_2,147,35). coords(lb2gt_2,20,53,35).  visible(lb2gt_2,35). orientation(lb2gt_3,146,35). coords(lb2gt_3,23,40,35).  visible(lb2gt_3,35).
orientation(lb2gt_2,147,36). coords(lb2gt_2,20,53,36).  visible(lb2gt_2,36). orientation(lb2gt_3,146,36). coords(lb2gt_3,23,40,36).  visible(lb2gt_3,36).
orientation(lb2gt_2,147,37). coords(lb2gt_2,20,53,37).  visible(lb2gt_2,37). orientation(lb2gt_3,146,37). coords(lb2gt_3,23,40,37).  visible(lb2gt_3,37).
orientation(lb2gt_2,147,38). coords(lb2gt_2,20,54,38).  visible(lb2gt_2,38). orientation(lb2gt_3,146,38). coords(lb2gt_3,21,39,38).  visible(lb2gt_3,38).
orientation(lb2gt_2,147,39). coords(lb2gt_2,20,53,39).  visible(lb2gt_2,39). orientation(lb2gt_3,146,39). coords(lb2gt_3,21,40,39).  visible(lb2gt_3,39).
orientation(lb2gt_2,147,40). coords(lb2gt_2,20,53,40).  visible(lb2gt_2,40). orientation(lb2gt_3,146,40). coords(lb2gt_3,20,40,40).  visible(lb2gt_3,40).
orientation(lb2gt_2,147,41). coords(lb2gt_2,20,53,41).  visible(lb2gt_2,41). orientation(lb2gt_3,146,41). coords(lb2gt_3,20,40,41).  visible(lb2gt_3,41).
orientation(lb2gt_2,147,42). coords(lb2gt_2,20,54,42).  visible(lb2gt_2,42). orientation(lb2gt_3,146,42). coords(lb2gt_3,20,40,42).  visible(lb2gt_3,42).
orientation(lb2gt_2,147,43). coords(lb2gt_2,19,54,43).  visible(lb2gt_2,43). orientation(lb2gt_3,146,43). coords(lb2gt_3,20,40,43).  visible(lb2gt_3,43).
orientation(lb2gt_2,147,44). coords(lb2gt_2,19,54,44).  visible(lb2gt_2,44). orientation(lb2gt_3,146,44). coords(lb2gt_3,20,40,44).  visible(lb2gt_3,44).
orientation(lb2gt_2,147,45). coords(lb2gt_2,19,54,45).  visible(lb2gt_2,45). orientation(lb2gt_3,146,45). coords(lb2gt_3,20,40,45).  visible(lb2gt_3,45).
orientation(lb2gt_2,147,46). coords(lb2gt_2,19,54,46).  visible(lb2gt_2,46). orientation(lb2gt_3,146,46). coords(lb2gt_3,20,40,46).  visible(lb2gt_3,46).
orientation(lb2gt_2,147,47). coords(lb2gt_2,19,54,47).  visible(lb2gt_2,47). orientation(lb2gt_3,146,47). coords(lb2gt_3,20,40,47).  visible(lb2gt_3,47).
orientation(lb2gt_2,147,48). coords(lb2gt_2,19,54,48).  visible(lb2gt_2,48). orientation(lb2gt_3,146,48). coords(lb2gt_3,19,40,48).  visible(lb2gt_3,48).
orientation(lb2gt_2,147,49). coords(lb2gt_2,19,54,49).  visible(lb2gt_2,49). orientation(lb2gt_3,146,49). coords(lb2gt_3,19,40,49).  visible(lb2gt_3,49).
happensAt(disappear(lb2gt_2),48). happensAt(disappear(lb2gt_3),48).
1{simpleEvent(T,lb2gt_2,active); simpleEvent(T,lb2gt_2,inactive); simpleEvent(T,lb2gt_2,walking); simpleEvent(T,lb2gt_2,running)}1 :- T=24..48.
1{simpleEvent(T,lb2gt_3,active); simpleEvent(T,lb2gt_3,inactive); simpleEvent(T,lb2gt_3,walking); simpleEvent(T,lb2gt_3,running)}1 :- T=24..48.
:- holdsAt(moving(lb2gt_2,lb2gt_3),24). :- holdsAt(moving(lb2gt_3,lb2gt_2),24).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),25). :- not holdsAt(moving(lb2gt_3,lb2gt_2),25).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),26). :- not holdsAt(moving(lb2gt_3,lb2gt_2),26).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),27). :- not holdsAt(moving(lb2gt_3,lb2gt_2),27).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),28). :- not holdsAt(moving(lb2gt_3,lb2gt_2),28).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),29). :- not holdsAt(moving(lb2gt_3,lb2gt_2),29).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),30). :- not holdsAt(moving(lb2gt_3,lb2gt_2),30).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),31). :- not holdsAt(moving(lb2gt_3,lb2gt_2),31).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),32). :- not holdsAt(moving(lb2gt_3,lb2gt_2),32).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),33). :- not holdsAt(moving(lb2gt_3,lb2gt_2),33).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),34). :- not holdsAt(moving(lb2gt_3,lb2gt_2),34).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),35). :- not holdsAt(moving(lb2gt_3,lb2gt_2),35).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),36). :- not holdsAt(moving(lb2gt_3,lb2gt_2),36).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),37). :- not holdsAt(moving(lb2gt_3,lb2gt_2),37).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),38). :- not holdsAt(moving(lb2gt_3,lb2gt_2),38).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),39). :- not holdsAt(moving(lb2gt_3,lb2gt_2),39).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),40). :- not holdsAt(moving(lb2gt_3,lb2gt_2),40).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),41). :- not holdsAt(moving(lb2gt_3,lb2gt_2),41).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),42). :- not holdsAt(moving(lb2gt_3,lb2gt_2),42).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),43). :- not holdsAt(moving(lb2gt_3,lb2gt_2),43).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),44). :- not holdsAt(moving(lb2gt_3,lb2gt_2),44).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),45). :- not holdsAt(moving(lb2gt_3,lb2gt_2),45).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),46). :- not holdsAt(moving(lb2gt_3,lb2gt_2),46).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),47). :- not holdsAt(moving(lb2gt_3,lb2gt_2),47).
:- not holdsAt(moving(lb2gt_2,lb2gt_3),48). :- not holdsAt(moving(lb2gt_3,lb2gt_2),48).
:- holdsAt(moving(lb2gt_2,lb2gt_3),49). :- holdsAt(moving(lb2gt_3,lb2gt_2),49).
:- simpleEvent(T,P,inactive), T>24 , T<48.
:- simpleEvent(T,P,running), T>24 , T<48.

#show simpleEvent/3.