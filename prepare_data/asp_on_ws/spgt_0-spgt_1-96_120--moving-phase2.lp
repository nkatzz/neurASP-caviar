

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
orientation(spgt_0,152,96). coords(spgt_0,70,150,96).  visible(spgt_0,96). orientation(spgt_1,154,96). coords(spgt_1,72,140,96).  visible(spgt_1,96).
orientation(spgt_0,152,97). coords(spgt_0,70,149,97).  visible(spgt_0,97). orientation(spgt_1,154,97). coords(spgt_1,72,139,97).  visible(spgt_1,97).
orientation(spgt_0,152,98). coords(spgt_0,70,149,98).  visible(spgt_0,98). orientation(spgt_1,154,98). coords(spgt_1,72,138,98).  visible(spgt_1,98).
orientation(spgt_0,152,99). coords(spgt_0,70,148,99).  visible(spgt_0,99). orientation(spgt_1,154,99). coords(spgt_1,72,138,99).  visible(spgt_1,99).
orientation(spgt_0,152,100). coords(spgt_0,70,148,100).  visible(spgt_0,100). orientation(spgt_1,154,100). coords(spgt_1,72,138,100).  visible(spgt_1,100).
orientation(spgt_0,152,101). coords(spgt_0,73,148,101).  visible(spgt_0,101). orientation(spgt_1,154,101). coords(spgt_1,72,137,101).  visible(spgt_1,101).
orientation(spgt_0,152,102). coords(spgt_0,74,148,102).  visible(spgt_0,102). orientation(spgt_1,154,102). coords(spgt_1,73,137,102).  visible(spgt_1,102).
orientation(spgt_0,152,103). coords(spgt_0,75,148,103).  visible(spgt_0,103). orientation(spgt_1,154,103). coords(spgt_1,75,137,103).  visible(spgt_1,103).
orientation(spgt_0,152,104). coords(spgt_0,76,147,104).  visible(spgt_0,104). orientation(spgt_1,154,104). coords(spgt_1,76,136,104).  visible(spgt_1,104).
orientation(spgt_0,152,105). coords(spgt_0,77,147,105).  visible(spgt_0,105). orientation(spgt_1,154,105). coords(spgt_1,77,136,105).  visible(spgt_1,105).
orientation(spgt_0,152,106). coords(spgt_0,77,147,106).  visible(spgt_0,106). orientation(spgt_1,154,106). coords(spgt_1,78,135,106).  visible(spgt_1,106).
orientation(spgt_0,152,107). coords(spgt_0,77,147,107).  visible(spgt_0,107). orientation(spgt_1,154,107). coords(spgt_1,78,134,107).  visible(spgt_1,107).
orientation(spgt_0,152,108). coords(spgt_0,77,147,108).  visible(spgt_0,108). orientation(spgt_1,154,108). coords(spgt_1,78,134,108).  visible(spgt_1,108).
orientation(spgt_0,152,109). coords(spgt_0,77,146,109).  visible(spgt_0,109). orientation(spgt_1,154,109). coords(spgt_1,78,134,109).  visible(spgt_1,109).
orientation(spgt_0,152,110). coords(spgt_0,78,145,110).  visible(spgt_0,110). orientation(spgt_1,154,110). coords(spgt_1,78,132,110).  visible(spgt_1,110).
orientation(spgt_0,152,111). coords(spgt_0,78,145,111).  visible(spgt_0,111). orientation(spgt_1,154,111). coords(spgt_1,78,132,111).  visible(spgt_1,111).
orientation(spgt_0,152,112). coords(spgt_0,78,144,112).  visible(spgt_0,112). orientation(spgt_1,154,112). coords(spgt_1,79,131,112).  visible(spgt_1,112).
orientation(spgt_0,152,113). coords(spgt_0,79,144,113).  visible(spgt_0,113). orientation(spgt_1,154,113). coords(spgt_1,79,130,113).  visible(spgt_1,113).
orientation(spgt_0,152,114). coords(spgt_0,79,143,114).  visible(spgt_0,114). orientation(spgt_1,154,114). coords(spgt_1,79,129,114).  visible(spgt_1,114).
orientation(spgt_0,152,115). coords(spgt_0,82,142,115).  visible(spgt_0,115). orientation(spgt_1,154,115). coords(spgt_1,81,128,115).  visible(spgt_1,115).
orientation(spgt_0,152,116). coords(spgt_0,82,142,116).  visible(spgt_0,116). orientation(spgt_1,154,116). coords(spgt_1,82,128,116).  visible(spgt_1,116).
orientation(spgt_0,152,117). coords(spgt_0,83,142,117).  visible(spgt_0,117). orientation(spgt_1,154,117). coords(spgt_1,82,128,117).  visible(spgt_1,117).
orientation(spgt_0,152,118). coords(spgt_0,84,141,118).  visible(spgt_0,118). orientation(spgt_1,154,118). coords(spgt_1,84,127,118).  visible(spgt_1,118).
orientation(spgt_0,152,119). coords(spgt_0,85,141,119).  visible(spgt_0,119). orientation(spgt_1,154,119). coords(spgt_1,84,127,119).  visible(spgt_1,119).
orientation(spgt_0,152,120). coords(spgt_0,85,140,120).  visible(spgt_0,120). orientation(spgt_1,154,120). coords(spgt_1,84,127,120).  visible(spgt_1,120).
orientation(spgt_0,152,121). coords(spgt_0,85,140,121).  visible(spgt_0,121). orientation(spgt_1,154,121). coords(spgt_1,84,127,121).  visible(spgt_1,121).
happensAt(disappear(spgt_0),120). happensAt(disappear(spgt_1),120).
1{simpleEvent(T,spgt_0,active); simpleEvent(T,spgt_0,inactive); simpleEvent(T,spgt_0,walking); simpleEvent(T,spgt_0,running)}1 :- T=96..120.
1{simpleEvent(T,spgt_1,active); simpleEvent(T,spgt_1,inactive); simpleEvent(T,spgt_1,walking); simpleEvent(T,spgt_1,running)}1 :- T=96..120.
:- holdsAt(moving(spgt_0,spgt_1),96). :- holdsAt(moving(spgt_1,spgt_0),96).
:- not holdsAt(moving(spgt_0,spgt_1),97). :- not holdsAt(moving(spgt_1,spgt_0),97).
:- not holdsAt(moving(spgt_0,spgt_1),98). :- not holdsAt(moving(spgt_1,spgt_0),98).
:- not holdsAt(moving(spgt_0,spgt_1),99). :- not holdsAt(moving(spgt_1,spgt_0),99).
:- not holdsAt(moving(spgt_0,spgt_1),100). :- not holdsAt(moving(spgt_1,spgt_0),100).
:- not holdsAt(moving(spgt_0,spgt_1),101). :- not holdsAt(moving(spgt_1,spgt_0),101).
:- not holdsAt(moving(spgt_0,spgt_1),102). :- not holdsAt(moving(spgt_1,spgt_0),102).
:- not holdsAt(moving(spgt_0,spgt_1),103). :- not holdsAt(moving(spgt_1,spgt_0),103).
:- not holdsAt(moving(spgt_0,spgt_1),104). :- not holdsAt(moving(spgt_1,spgt_0),104).
:- not holdsAt(moving(spgt_0,spgt_1),105). :- not holdsAt(moving(spgt_1,spgt_0),105).
:- not holdsAt(moving(spgt_0,spgt_1),106). :- not holdsAt(moving(spgt_1,spgt_0),106).
:- not holdsAt(moving(spgt_0,spgt_1),107). :- not holdsAt(moving(spgt_1,spgt_0),107).
:- not holdsAt(moving(spgt_0,spgt_1),108). :- not holdsAt(moving(spgt_1,spgt_0),108).
:- not holdsAt(moving(spgt_0,spgt_1),109). :- not holdsAt(moving(spgt_1,spgt_0),109).
:- not holdsAt(moving(spgt_0,spgt_1),110). :- not holdsAt(moving(spgt_1,spgt_0),110).
:- not holdsAt(moving(spgt_0,spgt_1),111). :- not holdsAt(moving(spgt_1,spgt_0),111).
:- not holdsAt(moving(spgt_0,spgt_1),112). :- not holdsAt(moving(spgt_1,spgt_0),112).
:- not holdsAt(moving(spgt_0,spgt_1),113). :- not holdsAt(moving(spgt_1,spgt_0),113).
:- not holdsAt(moving(spgt_0,spgt_1),114). :- not holdsAt(moving(spgt_1,spgt_0),114).
:- not holdsAt(moving(spgt_0,spgt_1),115). :- not holdsAt(moving(spgt_1,spgt_0),115).
:- not holdsAt(moving(spgt_0,spgt_1),116). :- not holdsAt(moving(spgt_1,spgt_0),116).
:- not holdsAt(moving(spgt_0,spgt_1),117). :- not holdsAt(moving(spgt_1,spgt_0),117).
:- not holdsAt(moving(spgt_0,spgt_1),118). :- not holdsAt(moving(spgt_1,spgt_0),118).
:- not holdsAt(moving(spgt_0,spgt_1),119). :- not holdsAt(moving(spgt_1,spgt_0),119).
:- not holdsAt(moving(spgt_0,spgt_1),120). :- not holdsAt(moving(spgt_1,spgt_0),120).
:- holdsAt(moving(spgt_0,spgt_1),121). :- holdsAt(moving(spgt_1,spgt_0),121).
:- simpleEvent(T,P,inactive), T>96 , T<120.
:- simpleEvent(T,P,running), T>96 , T<120.

#show simpleEvent/3.