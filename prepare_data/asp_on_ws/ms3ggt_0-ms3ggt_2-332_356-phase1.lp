
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

	orientation(ms3ggt_0,163,332). coords(ms3ggt_0,103,145,332).  visible(ms3ggt_0,332). happensAt(walking(ms3ggt_0),332). orientation(ms3ggt_2,13,332). coords(ms3ggt_2,214,273,332).  visible(ms3ggt_2,332). happensAt(walking(ms3ggt_2),332).
orientation(ms3ggt_0,163,333). coords(ms3ggt_0,104,145,333).  visible(ms3ggt_0,333). happensAt(walking(ms3ggt_0),333). orientation(ms3ggt_2,13,333). coords(ms3ggt_2,213,271,333).  visible(ms3ggt_2,333). happensAt(walking(ms3ggt_2),333).
orientation(ms3ggt_0,163,334). coords(ms3ggt_0,105,145,334).  visible(ms3ggt_0,334). happensAt(walking(ms3ggt_0),334). orientation(ms3ggt_2,13,334). coords(ms3ggt_2,213,269,334).  visible(ms3ggt_2,334). happensAt(walking(ms3ggt_2),334).
orientation(ms3ggt_0,163,335). coords(ms3ggt_0,106,145,335).  visible(ms3ggt_0,335). happensAt(walking(ms3ggt_0),335). orientation(ms3ggt_2,13,335). coords(ms3ggt_2,213,265,335).  visible(ms3ggt_2,335). happensAt(walking(ms3ggt_2),335).
orientation(ms3ggt_0,163,336). coords(ms3ggt_0,106,145,336).  visible(ms3ggt_0,336). happensAt(walking(ms3ggt_0),336). orientation(ms3ggt_2,13,336). coords(ms3ggt_2,213,264,336).  visible(ms3ggt_2,336). happensAt(walking(ms3ggt_2),336).
orientation(ms3ggt_0,163,337). coords(ms3ggt_0,106,145,337).  visible(ms3ggt_0,337). happensAt(walking(ms3ggt_0),337). orientation(ms3ggt_2,13,337). coords(ms3ggt_2,213,262,337).  visible(ms3ggt_2,337). happensAt(walking(ms3ggt_2),337).
orientation(ms3ggt_0,163,338). coords(ms3ggt_0,107,145,338).  visible(ms3ggt_0,338). happensAt(walking(ms3ggt_0),338). orientation(ms3ggt_2,13,338). coords(ms3ggt_2,212,260,338).  visible(ms3ggt_2,338). happensAt(walking(ms3ggt_2),338).
orientation(ms3ggt_0,156,339). coords(ms3ggt_0,107,144,339).  visible(ms3ggt_0,339). happensAt(walking(ms3ggt_0),339). orientation(ms3ggt_2,13,339). coords(ms3ggt_2,212,260,339).  visible(ms3ggt_2,339). happensAt(walking(ms3ggt_2),339).
orientation(ms3ggt_0,156,340). coords(ms3ggt_0,108,144,340).  visible(ms3ggt_0,340). happensAt(walking(ms3ggt_0),340). orientation(ms3ggt_2,13,340). coords(ms3ggt_2,209,258,340).  visible(ms3ggt_2,340). happensAt(walking(ms3ggt_2),340).
orientation(ms3ggt_0,156,341). coords(ms3ggt_0,108,144,341).  visible(ms3ggt_0,341). happensAt(walking(ms3ggt_0),341). orientation(ms3ggt_2,13,341). coords(ms3ggt_2,208,256,341).  visible(ms3ggt_2,341). happensAt(walking(ms3ggt_2),341).
orientation(ms3ggt_0,156,342). coords(ms3ggt_0,109,144,342).  visible(ms3ggt_0,342). happensAt(walking(ms3ggt_0),342). orientation(ms3ggt_2,13,342). coords(ms3ggt_2,209,254,342).  visible(ms3ggt_2,342). happensAt(walking(ms3ggt_2),342).
orientation(ms3ggt_0,156,343). coords(ms3ggt_0,109,144,343).  visible(ms3ggt_0,343). happensAt(walking(ms3ggt_0),343). orientation(ms3ggt_2,13,343). coords(ms3ggt_2,205,253,343).  visible(ms3ggt_2,343). happensAt(walking(ms3ggt_2),343).
orientation(ms3ggt_0,158,344). coords(ms3ggt_0,112,144,344).  visible(ms3ggt_0,344). happensAt(walking(ms3ggt_0),344). orientation(ms3ggt_2,13,344). coords(ms3ggt_2,204,250,344).  visible(ms3ggt_2,344). happensAt(walking(ms3ggt_2),344).
orientation(ms3ggt_0,152,345). coords(ms3ggt_0,114,144,345).  visible(ms3ggt_0,345). happensAt(walking(ms3ggt_0),345). orientation(ms3ggt_2,13,345). coords(ms3ggt_2,204,250,345).  visible(ms3ggt_2,345). happensAt(walking(ms3ggt_2),345).
orientation(ms3ggt_0,152,346). coords(ms3ggt_0,116,144,346).  visible(ms3ggt_0,346). happensAt(walking(ms3ggt_0),346). orientation(ms3ggt_2,13,346). coords(ms3ggt_2,200,249,346).  visible(ms3ggt_2,346). happensAt(walking(ms3ggt_2),346).
orientation(ms3ggt_0,152,347). coords(ms3ggt_0,117,144,347).  visible(ms3ggt_0,347). happensAt(walking(ms3ggt_0),347). orientation(ms3ggt_2,8,347). coords(ms3ggt_2,198,244,347).  visible(ms3ggt_2,347). happensAt(walking(ms3ggt_2),347).
orientation(ms3ggt_0,157,348). coords(ms3ggt_0,118,144,348).  visible(ms3ggt_0,348). happensAt(walking(ms3ggt_0),348). orientation(ms3ggt_2,8,348). coords(ms3ggt_2,197,241,348).  visible(ms3ggt_2,348). happensAt(walking(ms3ggt_2),348).
orientation(ms3ggt_0,158,349). coords(ms3ggt_0,119,144,349).  visible(ms3ggt_0,349). happensAt(walking(ms3ggt_0),349). orientation(ms3ggt_2,8,349). coords(ms3ggt_2,197,240,349).  visible(ms3ggt_2,349). happensAt(walking(ms3ggt_2),349).
orientation(ms3ggt_0,158,350). coords(ms3ggt_0,120,145,350).  visible(ms3ggt_0,350). happensAt(walking(ms3ggt_0),350). orientation(ms3ggt_2,8,350). coords(ms3ggt_2,197,238,350).  visible(ms3ggt_2,350). happensAt(walking(ms3ggt_2),350).
orientation(ms3ggt_0,160,351). coords(ms3ggt_0,121,145,351).  visible(ms3ggt_0,351). happensAt(walking(ms3ggt_0),351). orientation(ms3ggt_2,8,351). coords(ms3ggt_2,196,238,351).  visible(ms3ggt_2,351). happensAt(walking(ms3ggt_2),351).
orientation(ms3ggt_0,160,352). coords(ms3ggt_0,121,145,352).  visible(ms3ggt_0,352). happensAt(walking(ms3ggt_0),352). orientation(ms3ggt_2,8,352). coords(ms3ggt_2,196,238,352).  visible(ms3ggt_2,352). happensAt(walking(ms3ggt_2),352).
orientation(ms3ggt_0,160,353). coords(ms3ggt_0,121,145,353).  visible(ms3ggt_0,353). happensAt(walking(ms3ggt_0),353). orientation(ms3ggt_2,8,353). coords(ms3ggt_2,195,237,353).  visible(ms3ggt_2,353). happensAt(walking(ms3ggt_2),353).
orientation(ms3ggt_0,160,354). coords(ms3ggt_0,121,145,354).  visible(ms3ggt_0,354). happensAt(walking(ms3ggt_0),354). orientation(ms3ggt_2,8,354). coords(ms3ggt_2,194,236,354).  visible(ms3ggt_2,354). happensAt(walking(ms3ggt_2),354).
orientation(ms3ggt_0,157,355). coords(ms3ggt_0,121,144,355).  visible(ms3ggt_0,355). happensAt(walking(ms3ggt_0),355). orientation(ms3ggt_2,8,355). coords(ms3ggt_2,191,234,355).  visible(ms3ggt_2,355). happensAt(walking(ms3ggt_2),355).
orientation(ms3ggt_0,157,356). coords(ms3ggt_0,122,144,356).  visible(ms3ggt_0,356). happensAt(walking(ms3ggt_0),356). orientation(ms3ggt_2,8,356). coords(ms3ggt_2,191,233,356).  visible(ms3ggt_2,356). happensAt(walking(ms3ggt_2),356).
#show holdsAt/2.