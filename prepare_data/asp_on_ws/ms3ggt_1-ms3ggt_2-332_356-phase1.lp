
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


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(ms3ggt_1,156,332). coords(ms3ggt_1,97,167,332).  visible(ms3ggt_1,332). happensAt(walking(ms3ggt_1),332). orientation(ms3ggt_2,13,332). coords(ms3ggt_2,214,273,332).  visible(ms3ggt_2,332). happensAt(walking(ms3ggt_2),332).
orientation(ms3ggt_1,156,333). coords(ms3ggt_1,98,167,333).  visible(ms3ggt_1,333). happensAt(walking(ms3ggt_1),333). orientation(ms3ggt_2,13,333). coords(ms3ggt_2,213,271,333).  visible(ms3ggt_2,333). happensAt(walking(ms3ggt_2),333).
orientation(ms3ggt_1,156,334). coords(ms3ggt_1,100,167,334).  visible(ms3ggt_1,334). happensAt(walking(ms3ggt_1),334). orientation(ms3ggt_2,13,334). coords(ms3ggt_2,213,269,334).  visible(ms3ggt_2,334). happensAt(walking(ms3ggt_2),334).
orientation(ms3ggt_1,156,335). coords(ms3ggt_1,101,166,335).  visible(ms3ggt_1,335). happensAt(walking(ms3ggt_1),335). orientation(ms3ggt_2,13,335). coords(ms3ggt_2,213,265,335).  visible(ms3ggt_2,335). happensAt(walking(ms3ggt_2),335).
orientation(ms3ggt_1,156,336). coords(ms3ggt_1,103,166,336).  visible(ms3ggt_1,336). happensAt(walking(ms3ggt_1),336). orientation(ms3ggt_2,13,336). coords(ms3ggt_2,213,264,336).  visible(ms3ggt_2,336). happensAt(walking(ms3ggt_2),336).
orientation(ms3ggt_1,156,337). coords(ms3ggt_1,103,166,337).  visible(ms3ggt_1,337). happensAt(walking(ms3ggt_1),337). orientation(ms3ggt_2,13,337). coords(ms3ggt_2,213,262,337).  visible(ms3ggt_2,337). happensAt(walking(ms3ggt_2),337).
orientation(ms3ggt_1,156,338). coords(ms3ggt_1,104,166,338).  visible(ms3ggt_1,338). happensAt(walking(ms3ggt_1),338). orientation(ms3ggt_2,13,338). coords(ms3ggt_2,212,260,338).  visible(ms3ggt_2,338). happensAt(walking(ms3ggt_2),338).
orientation(ms3ggt_1,156,339). coords(ms3ggt_1,104,166,339).  visible(ms3ggt_1,339). happensAt(walking(ms3ggt_1),339). orientation(ms3ggt_2,13,339). coords(ms3ggt_2,212,260,339).  visible(ms3ggt_2,339). happensAt(walking(ms3ggt_2),339).
orientation(ms3ggt_1,156,340). coords(ms3ggt_1,105,166,340).  visible(ms3ggt_1,340). happensAt(walking(ms3ggt_1),340). orientation(ms3ggt_2,13,340). coords(ms3ggt_2,209,258,340).  visible(ms3ggt_2,340). happensAt(walking(ms3ggt_2),340).
orientation(ms3ggt_1,156,341). coords(ms3ggt_1,105,166,341).  visible(ms3ggt_1,341). happensAt(walking(ms3ggt_1),341). orientation(ms3ggt_2,13,341). coords(ms3ggt_2,208,256,341).  visible(ms3ggt_2,341). happensAt(walking(ms3ggt_2),341).
orientation(ms3ggt_1,156,342). coords(ms3ggt_1,105,166,342).  visible(ms3ggt_1,342). happensAt(walking(ms3ggt_1),342). orientation(ms3ggt_2,13,342). coords(ms3ggt_2,209,254,342).  visible(ms3ggt_2,342). happensAt(walking(ms3ggt_2),342).
orientation(ms3ggt_1,156,343). coords(ms3ggt_1,105,166,343).  visible(ms3ggt_1,343). happensAt(walking(ms3ggt_1),343). orientation(ms3ggt_2,13,343). coords(ms3ggt_2,205,253,343).  visible(ms3ggt_2,343). happensAt(walking(ms3ggt_2),343).
orientation(ms3ggt_1,156,344). coords(ms3ggt_1,106,166,344).  visible(ms3ggt_1,344). happensAt(walking(ms3ggt_1),344). orientation(ms3ggt_2,13,344). coords(ms3ggt_2,204,250,344).  visible(ms3ggt_2,344). happensAt(walking(ms3ggt_2),344).
orientation(ms3ggt_1,156,345). coords(ms3ggt_1,108,165,345).  visible(ms3ggt_1,345). happensAt(walking(ms3ggt_1),345). orientation(ms3ggt_2,13,345). coords(ms3ggt_2,204,250,345).  visible(ms3ggt_2,345). happensAt(walking(ms3ggt_2),345).
orientation(ms3ggt_1,156,346). coords(ms3ggt_1,110,165,346).  visible(ms3ggt_1,346). happensAt(walking(ms3ggt_1),346). orientation(ms3ggt_2,13,346). coords(ms3ggt_2,200,249,346).  visible(ms3ggt_2,346). happensAt(walking(ms3ggt_2),346).
orientation(ms3ggt_1,156,347). coords(ms3ggt_1,111,164,347).  visible(ms3ggt_1,347). happensAt(walking(ms3ggt_1),347). orientation(ms3ggt_2,8,347). coords(ms3ggt_2,198,244,347).  visible(ms3ggt_2,347). happensAt(walking(ms3ggt_2),347).
orientation(ms3ggt_1,156,348). coords(ms3ggt_1,112,164,348).  visible(ms3ggt_1,348). happensAt(walking(ms3ggt_1),348). orientation(ms3ggt_2,8,348). coords(ms3ggt_2,197,241,348).  visible(ms3ggt_2,348). happensAt(walking(ms3ggt_2),348).
orientation(ms3ggt_1,156,349). coords(ms3ggt_1,113,164,349).  visible(ms3ggt_1,349). happensAt(walking(ms3ggt_1),349). orientation(ms3ggt_2,8,349). coords(ms3ggt_2,197,240,349).  visible(ms3ggt_2,349). happensAt(walking(ms3ggt_2),349).
orientation(ms3ggt_1,156,350). coords(ms3ggt_1,114,164,350).  visible(ms3ggt_1,350). happensAt(walking(ms3ggt_1),350). orientation(ms3ggt_2,8,350). coords(ms3ggt_2,197,238,350).  visible(ms3ggt_2,350). happensAt(walking(ms3ggt_2),350).
orientation(ms3ggt_1,156,351). coords(ms3ggt_1,115,164,351).  visible(ms3ggt_1,351). happensAt(walking(ms3ggt_1),351). orientation(ms3ggt_2,8,351). coords(ms3ggt_2,196,238,351).  visible(ms3ggt_2,351). happensAt(walking(ms3ggt_2),351).
orientation(ms3ggt_1,156,352). coords(ms3ggt_1,115,163,352).  visible(ms3ggt_1,352). happensAt(walking(ms3ggt_1),352). orientation(ms3ggt_2,8,352). coords(ms3ggt_2,196,238,352).  visible(ms3ggt_2,352). happensAt(walking(ms3ggt_2),352).
orientation(ms3ggt_1,156,353). coords(ms3ggt_1,115,163,353).  visible(ms3ggt_1,353). happensAt(walking(ms3ggt_1),353). orientation(ms3ggt_2,8,353). coords(ms3ggt_2,195,237,353).  visible(ms3ggt_2,353). happensAt(walking(ms3ggt_2),353).
orientation(ms3ggt_1,156,354). coords(ms3ggt_1,115,163,354).  visible(ms3ggt_1,354). happensAt(walking(ms3ggt_1),354). orientation(ms3ggt_2,8,354). coords(ms3ggt_2,194,236,354).  visible(ms3ggt_2,354). happensAt(walking(ms3ggt_2),354).
orientation(ms3ggt_1,156,355). coords(ms3ggt_1,116,162,355).  visible(ms3ggt_1,355). happensAt(walking(ms3ggt_1),355). orientation(ms3ggt_2,8,355). coords(ms3ggt_2,191,234,355).  visible(ms3ggt_2,355). happensAt(walking(ms3ggt_2),355).
orientation(ms3ggt_1,156,356). coords(ms3ggt_1,116,162,356).  visible(ms3ggt_1,356). happensAt(walking(ms3ggt_1),356). orientation(ms3ggt_2,8,356). coords(ms3ggt_2,191,233,356).  visible(ms3ggt_2,356). happensAt(walking(ms3ggt_2),356).
#show holdsAt/2.