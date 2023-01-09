
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

	orientation(lbpugt_2,84,839). coords(lbpugt_2,292,96,839).  visible(lbpugt_2,839). happensAt(active(lbpugt_2),839). orientation(lbpugt_4,135,839). coords(lbpugt_4,219,34,839).  visible(lbpugt_4,839). happensAt(inactive(lbpugt_4),839).
orientation(lbpugt_2,84,840). coords(lbpugt_2,292,96,840).  visible(lbpugt_2,840). happensAt(active(lbpugt_2),840). orientation(lbpugt_4,135,840). coords(lbpugt_4,219,34,840).  visible(lbpugt_4,840). happensAt(inactive(lbpugt_4),840).
orientation(lbpugt_2,84,841). coords(lbpugt_2,292,96,841).  visible(lbpugt_2,841). happensAt(active(lbpugt_2),841). orientation(lbpugt_4,135,841). coords(lbpugt_4,219,34,841).  visible(lbpugt_4,841). happensAt(inactive(lbpugt_4),841).
orientation(lbpugt_2,84,842). coords(lbpugt_2,292,96,842).  visible(lbpugt_2,842). happensAt(active(lbpugt_2),842). orientation(lbpugt_4,135,842). coords(lbpugt_4,219,34,842).  visible(lbpugt_4,842). happensAt(inactive(lbpugt_4),842).
orientation(lbpugt_2,84,843). coords(lbpugt_2,292,96,843).  visible(lbpugt_2,843). happensAt(active(lbpugt_2),843). orientation(lbpugt_4,135,843). coords(lbpugt_4,219,34,843).  visible(lbpugt_4,843). happensAt(inactive(lbpugt_4),843).
orientation(lbpugt_2,84,844). coords(lbpugt_2,292,96,844).  visible(lbpugt_2,844). happensAt(active(lbpugt_2),844). orientation(lbpugt_4,135,844). coords(lbpugt_4,219,34,844).  visible(lbpugt_4,844). happensAt(inactive(lbpugt_4),844).
orientation(lbpugt_2,84,845). coords(lbpugt_2,292,96,845).  visible(lbpugt_2,845). happensAt(active(lbpugt_2),845). orientation(lbpugt_4,135,845). coords(lbpugt_4,219,34,845).  visible(lbpugt_4,845). happensAt(inactive(lbpugt_4),845).
orientation(lbpugt_2,84,846). coords(lbpugt_2,292,96,846).  visible(lbpugt_2,846). happensAt(active(lbpugt_2),846). orientation(lbpugt_4,135,846). coords(lbpugt_4,219,34,846).  visible(lbpugt_4,846). happensAt(inactive(lbpugt_4),846).
orientation(lbpugt_2,84,847). coords(lbpugt_2,292,96,847).  visible(lbpugt_2,847). happensAt(active(lbpugt_2),847). orientation(lbpugt_4,135,847). coords(lbpugt_4,219,34,847).  visible(lbpugt_4,847). happensAt(inactive(lbpugt_4),847).
orientation(lbpugt_2,84,848). coords(lbpugt_2,292,96,848).  visible(lbpugt_2,848). happensAt(active(lbpugt_2),848). orientation(lbpugt_4,135,848). coords(lbpugt_4,219,34,848).  visible(lbpugt_4,848). happensAt(inactive(lbpugt_4),848).
orientation(lbpugt_2,84,849). coords(lbpugt_2,292,96,849).  visible(lbpugt_2,849). happensAt(active(lbpugt_2),849). orientation(lbpugt_4,135,849). coords(lbpugt_4,219,34,849).  visible(lbpugt_4,849). happensAt(inactive(lbpugt_4),849).
orientation(lbpugt_2,84,850). coords(lbpugt_2,292,96,850).  visible(lbpugt_2,850). happensAt(active(lbpugt_2),850). orientation(lbpugt_4,135,850). coords(lbpugt_4,219,34,850).  visible(lbpugt_4,850). happensAt(inactive(lbpugt_4),850).
orientation(lbpugt_2,84,851). coords(lbpugt_2,292,95,851).  visible(lbpugt_2,851). happensAt(active(lbpugt_2),851). orientation(lbpugt_4,135,851). coords(lbpugt_4,219,34,851).  visible(lbpugt_4,851). happensAt(inactive(lbpugt_4),851).
orientation(lbpugt_2,84,852). coords(lbpugt_2,292,95,852).  visible(lbpugt_2,852). happensAt(active(lbpugt_2),852). orientation(lbpugt_4,135,852). coords(lbpugt_4,219,34,852).  visible(lbpugt_4,852). happensAt(inactive(lbpugt_4),852).
orientation(lbpugt_2,84,853). coords(lbpugt_2,292,95,853).  visible(lbpugt_2,853). happensAt(active(lbpugt_2),853). orientation(lbpugt_4,135,853). coords(lbpugt_4,219,34,853).  visible(lbpugt_4,853). happensAt(inactive(lbpugt_4),853).
orientation(lbpugt_2,84,854). coords(lbpugt_2,292,96,854).  visible(lbpugt_2,854). happensAt(active(lbpugt_2),854). orientation(lbpugt_4,135,854). coords(lbpugt_4,219,34,854).  visible(lbpugt_4,854). happensAt(inactive(lbpugt_4),854).
orientation(lbpugt_2,84,855). coords(lbpugt_2,292,96,855).  visible(lbpugt_2,855). happensAt(active(lbpugt_2),855). orientation(lbpugt_4,135,855). coords(lbpugt_4,219,34,855).  visible(lbpugt_4,855). happensAt(inactive(lbpugt_4),855).
orientation(lbpugt_2,84,856). coords(lbpugt_2,292,96,856).  visible(lbpugt_2,856). happensAt(active(lbpugt_2),856). orientation(lbpugt_4,135,856). coords(lbpugt_4,219,34,856).  visible(lbpugt_4,856). happensAt(inactive(lbpugt_4),856).
orientation(lbpugt_2,84,857). coords(lbpugt_2,292,96,857).  visible(lbpugt_2,857). happensAt(active(lbpugt_2),857). orientation(lbpugt_4,135,857). coords(lbpugt_4,219,34,857).  visible(lbpugt_4,857). happensAt(inactive(lbpugt_4),857).
orientation(lbpugt_2,84,858). coords(lbpugt_2,292,97,858).  visible(lbpugt_2,858). happensAt(active(lbpugt_2),858). orientation(lbpugt_4,135,858). coords(lbpugt_4,219,34,858).  visible(lbpugt_4,858). happensAt(inactive(lbpugt_4),858).
orientation(lbpugt_2,84,859). coords(lbpugt_2,292,98,859).  visible(lbpugt_2,859). happensAt(walking(lbpugt_2),859). orientation(lbpugt_4,135,859). coords(lbpugt_4,219,34,859).  visible(lbpugt_4,859). happensAt(inactive(lbpugt_4),859).
orientation(lbpugt_2,84,860). coords(lbpugt_2,291,98,860).  visible(lbpugt_2,860). happensAt(walking(lbpugt_2),860). orientation(lbpugt_4,135,860). coords(lbpugt_4,219,34,860).  visible(lbpugt_4,860). happensAt(inactive(lbpugt_4),860).
orientation(lbpugt_2,84,861). coords(lbpugt_2,290,99,861).  visible(lbpugt_2,861). happensAt(walking(lbpugt_2),861). orientation(lbpugt_4,135,861). coords(lbpugt_4,219,34,861).  visible(lbpugt_4,861). happensAt(inactive(lbpugt_4),861).
orientation(lbpugt_2,84,862). coords(lbpugt_2,290,100,862).  visible(lbpugt_2,862). happensAt(walking(lbpugt_2),862). orientation(lbpugt_4,135,862). coords(lbpugt_4,219,34,862).  visible(lbpugt_4,862). happensAt(inactive(lbpugt_4),862).
orientation(lbpugt_2,87,863). coords(lbpugt_2,290,101,863).  visible(lbpugt_2,863). happensAt(walking(lbpugt_2),863). orientation(lbpugt_4,135,863). coords(lbpugt_4,219,34,863).  visible(lbpugt_4,863). happensAt(inactive(lbpugt_4),863).
#show holdsAt/2.