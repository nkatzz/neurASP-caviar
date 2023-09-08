

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

	person(mc1gt_0;mc1gt_2).
orientation(mc1gt_0,148,105). coords(mc1gt_0,80,97,105).  visible(mc1gt_0,105). orientation(mc1gt_2,144,105). coords(mc1gt_2,76,68,105).  visible(mc1gt_2,105).
orientation(mc1gt_0,148,106). coords(mc1gt_0,80,98,106).  visible(mc1gt_0,106). orientation(mc1gt_2,144,106). coords(mc1gt_2,76,72,106).  visible(mc1gt_2,106).
orientation(mc1gt_0,148,107). coords(mc1gt_0,80,99,107).  visible(mc1gt_0,107). orientation(mc1gt_2,144,107). coords(mc1gt_2,76,73,107).  visible(mc1gt_2,107).
orientation(mc1gt_0,148,108). coords(mc1gt_0,80,99,108).  visible(mc1gt_0,108). orientation(mc1gt_2,144,108). coords(mc1gt_2,76,73,108).  visible(mc1gt_2,108).
orientation(mc1gt_0,147,109). coords(mc1gt_0,81,100,109).  visible(mc1gt_0,109). orientation(mc1gt_2,144,109). coords(mc1gt_2,76,74,109).  visible(mc1gt_2,109).
orientation(mc1gt_0,147,110). coords(mc1gt_0,81,101,110).  visible(mc1gt_0,110). orientation(mc1gt_2,144,110). coords(mc1gt_2,76,74,110).  visible(mc1gt_2,110).
orientation(mc1gt_0,147,111). coords(mc1gt_0,82,101,111).  visible(mc1gt_0,111). orientation(mc1gt_2,144,111). coords(mc1gt_2,77,74,111).  visible(mc1gt_2,111).
orientation(mc1gt_0,147,112). coords(mc1gt_0,82,101,112).  visible(mc1gt_0,112). orientation(mc1gt_2,144,112). coords(mc1gt_2,77,75,112).  visible(mc1gt_2,112).
orientation(mc1gt_0,147,113). coords(mc1gt_0,82,101,113).  visible(mc1gt_0,113). orientation(mc1gt_2,144,113). coords(mc1gt_2,76,75,113).  visible(mc1gt_2,113).
orientation(mc1gt_0,147,114). coords(mc1gt_0,82,101,114).  visible(mc1gt_0,114). orientation(mc1gt_2,144,114). coords(mc1gt_2,76,76,114).  visible(mc1gt_2,114).
orientation(mc1gt_0,147,115). coords(mc1gt_0,82,101,115).  visible(mc1gt_0,115). orientation(mc1gt_2,144,115). coords(mc1gt_2,76,77,115).  visible(mc1gt_2,115).
orientation(mc1gt_0,147,116). coords(mc1gt_0,82,101,116).  visible(mc1gt_0,116). orientation(mc1gt_2,144,116). coords(mc1gt_2,77,78,116).  visible(mc1gt_2,116).
orientation(mc1gt_0,147,117). coords(mc1gt_0,82,101,117).  visible(mc1gt_0,117). orientation(mc1gt_2,144,117). coords(mc1gt_2,77,78,117).  visible(mc1gt_2,117).
orientation(mc1gt_0,147,118). coords(mc1gt_0,83,102,118).  visible(mc1gt_0,118). orientation(mc1gt_2,144,118). coords(mc1gt_2,77,79,118).  visible(mc1gt_2,118).
orientation(mc1gt_0,147,119). coords(mc1gt_0,83,102,119).  visible(mc1gt_0,119). orientation(mc1gt_2,144,119). coords(mc1gt_2,77,79,119).  visible(mc1gt_2,119).
orientation(mc1gt_0,147,120). coords(mc1gt_0,84,102,120).  visible(mc1gt_0,120). orientation(mc1gt_2,144,120). coords(mc1gt_2,77,79,120).  visible(mc1gt_2,120).
orientation(mc1gt_0,147,121). coords(mc1gt_0,85,102,121).  visible(mc1gt_0,121). orientation(mc1gt_2,144,121). coords(mc1gt_2,77,80,121).  visible(mc1gt_2,121).
orientation(mc1gt_0,147,122). coords(mc1gt_0,86,102,122).  visible(mc1gt_0,122). orientation(mc1gt_2,144,122). coords(mc1gt_2,77,80,122).  visible(mc1gt_2,122).
orientation(mc1gt_0,147,123). coords(mc1gt_0,87,102,123).  visible(mc1gt_0,123). orientation(mc1gt_2,144,123). coords(mc1gt_2,77,81,123).  visible(mc1gt_2,123).
orientation(mc1gt_0,147,124). coords(mc1gt_0,87,102,124).  visible(mc1gt_0,124). orientation(mc1gt_2,144,124). coords(mc1gt_2,77,83,124).  visible(mc1gt_2,124).
orientation(mc1gt_0,147,125). coords(mc1gt_0,87,102,125).  visible(mc1gt_0,125). orientation(mc1gt_2,144,125). coords(mc1gt_2,76,85,125).  visible(mc1gt_2,125).
orientation(mc1gt_0,147,126). coords(mc1gt_0,88,103,126).  visible(mc1gt_0,126). orientation(mc1gt_2,144,126). coords(mc1gt_2,76,86,126).  visible(mc1gt_2,126).
orientation(mc1gt_0,147,127). coords(mc1gt_0,89,103,127).  visible(mc1gt_0,127). orientation(mc1gt_2,144,127). coords(mc1gt_2,76,86,127).  visible(mc1gt_2,127).
orientation(mc1gt_0,147,128). coords(mc1gt_0,89,103,128).  visible(mc1gt_0,128). orientation(mc1gt_2,144,128). coords(mc1gt_2,76,87,128).  visible(mc1gt_2,128).
orientation(mc1gt_0,147,129). coords(mc1gt_0,89,103,129).  visible(mc1gt_0,129). orientation(mc1gt_2,144,129). coords(mc1gt_2,76,87,129).  visible(mc1gt_2,129).
orientation(mc1gt_0,147,130). coords(mc1gt_0,89,103,130).  visible(mc1gt_0,130). orientation(mc1gt_2,144,130). coords(mc1gt_2,76,87,130).  visible(mc1gt_2,130).
happensAt(disappear(mc1gt_0),129). happensAt(disappear(mc1gt_2),129).
1{simpleEvent(T,mc1gt_0,active); simpleEvent(T,mc1gt_0,inactive); simpleEvent(T,mc1gt_0,walking); simpleEvent(T,mc1gt_0,running)}1 :- T=105..129.
1{simpleEvent(T,mc1gt_2,active); simpleEvent(T,mc1gt_2,inactive); simpleEvent(T,mc1gt_2,walking); simpleEvent(T,mc1gt_2,running)}1 :- T=105..129.
:- holdsAt(moving(mc1gt_0,mc1gt_2),105). :- holdsAt(moving(mc1gt_2,mc1gt_0),105).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),106). :- not holdsAt(moving(mc1gt_2,mc1gt_0),106).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),107). :- not holdsAt(moving(mc1gt_2,mc1gt_0),107).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),108). :- not holdsAt(moving(mc1gt_2,mc1gt_0),108).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),109). :- not holdsAt(moving(mc1gt_2,mc1gt_0),109).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),110). :- not holdsAt(moving(mc1gt_2,mc1gt_0),110).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),111). :- not holdsAt(moving(mc1gt_2,mc1gt_0),111).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),112). :- not holdsAt(moving(mc1gt_2,mc1gt_0),112).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),113). :- not holdsAt(moving(mc1gt_2,mc1gt_0),113).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),114). :- not holdsAt(moving(mc1gt_2,mc1gt_0),114).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),115). :- not holdsAt(moving(mc1gt_2,mc1gt_0),115).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),116). :- not holdsAt(moving(mc1gt_2,mc1gt_0),116).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),117). :- not holdsAt(moving(mc1gt_2,mc1gt_0),117).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),118). :- not holdsAt(moving(mc1gt_2,mc1gt_0),118).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),119). :- not holdsAt(moving(mc1gt_2,mc1gt_0),119).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),120). :- not holdsAt(moving(mc1gt_2,mc1gt_0),120).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),121). :- not holdsAt(moving(mc1gt_2,mc1gt_0),121).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),122). :- not holdsAt(moving(mc1gt_2,mc1gt_0),122).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),123). :- not holdsAt(moving(mc1gt_2,mc1gt_0),123).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),124). :- not holdsAt(moving(mc1gt_2,mc1gt_0),124).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),125). :- not holdsAt(moving(mc1gt_2,mc1gt_0),125).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),126). :- not holdsAt(moving(mc1gt_2,mc1gt_0),126).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),127). :- not holdsAt(moving(mc1gt_2,mc1gt_0),127).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),128). :- not holdsAt(moving(mc1gt_2,mc1gt_0),128).
:- not holdsAt(moving(mc1gt_0,mc1gt_2),129). :- not holdsAt(moving(mc1gt_2,mc1gt_0),129).
:- holdsAt(moving(mc1gt_0,mc1gt_2),130). :- holdsAt(moving(mc1gt_2,mc1gt_0),130).
:- simpleEvent(T,P,inactive), T>105 , T<129.
:- simpleEvent(T,P,running), T>105 , T<129.

#show simpleEvent/3.