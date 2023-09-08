
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

	orientation(mws1gt_1,155,260). coords(mws1gt_1,102,115,260).  visible(mws1gt_1,260). happensAt(walking(mws1gt_1),260). orientation(mws1gt_3,157,260). coords(mws1gt_3,93,139,260).  visible(mws1gt_3,260). happensAt(walking(mws1gt_3),260).
orientation(mws1gt_1,155,261). coords(mws1gt_1,102,114,261).  visible(mws1gt_1,261). happensAt(walking(mws1gt_1),261). orientation(mws1gt_3,157,261). coords(mws1gt_3,93,138,261).  visible(mws1gt_3,261). happensAt(walking(mws1gt_3),261).
orientation(mws1gt_1,155,262). coords(mws1gt_1,101,113,262).  visible(mws1gt_1,262). happensAt(walking(mws1gt_1),262). orientation(mws1gt_3,157,262). coords(mws1gt_3,93,137,262).  visible(mws1gt_3,262). happensAt(walking(mws1gt_3),262).
orientation(mws1gt_1,155,263). coords(mws1gt_1,101,113,263).  visible(mws1gt_1,263). happensAt(walking(mws1gt_1),263). orientation(mws1gt_3,157,263). coords(mws1gt_3,92,137,263).  visible(mws1gt_3,263). happensAt(walking(mws1gt_3),263).
orientation(mws1gt_1,155,264). coords(mws1gt_1,100,112,264).  visible(mws1gt_1,264). happensAt(walking(mws1gt_1),264). orientation(mws1gt_3,157,264). coords(mws1gt_3,92,137,264).  visible(mws1gt_3,264). happensAt(walking(mws1gt_3),264).
orientation(mws1gt_1,155,265). coords(mws1gt_1,100,111,265).  visible(mws1gt_1,265). happensAt(walking(mws1gt_1),265). orientation(mws1gt_3,157,265). coords(mws1gt_3,92,136,265).  visible(mws1gt_3,265). happensAt(walking(mws1gt_3),265).
orientation(mws1gt_1,155,266). coords(mws1gt_1,100,111,266).  visible(mws1gt_1,266). happensAt(walking(mws1gt_1),266). orientation(mws1gt_3,157,266). coords(mws1gt_3,91,136,266).  visible(mws1gt_3,266). happensAt(walking(mws1gt_3),266).
orientation(mws1gt_1,155,267). coords(mws1gt_1,100,111,267).  visible(mws1gt_1,267). happensAt(walking(mws1gt_1),267). orientation(mws1gt_3,157,267). coords(mws1gt_3,90,135,267).  visible(mws1gt_3,267). happensAt(walking(mws1gt_3),267).
orientation(mws1gt_1,155,268). coords(mws1gt_1,100,111,268).  visible(mws1gt_1,268). happensAt(walking(mws1gt_1),268). orientation(mws1gt_3,157,268). coords(mws1gt_3,89,134,268).  visible(mws1gt_3,268). happensAt(walking(mws1gt_3),268).
orientation(mws1gt_1,155,269). coords(mws1gt_1,99,111,269).  visible(mws1gt_1,269). happensAt(walking(mws1gt_1),269). orientation(mws1gt_3,157,269). coords(mws1gt_3,88,134,269).  visible(mws1gt_3,269). happensAt(walking(mws1gt_3),269).
orientation(mws1gt_1,155,270). coords(mws1gt_1,99,111,270).  visible(mws1gt_1,270). happensAt(walking(mws1gt_1),270). orientation(mws1gt_3,157,270). coords(mws1gt_3,88,134,270).  visible(mws1gt_3,270). happensAt(walking(mws1gt_3),270).
orientation(mws1gt_1,155,271). coords(mws1gt_1,98,111,271).  visible(mws1gt_1,271). happensAt(walking(mws1gt_1),271). orientation(mws1gt_3,157,271). coords(mws1gt_3,88,133,271).  visible(mws1gt_3,271). happensAt(walking(mws1gt_3),271).
orientation(mws1gt_1,155,272). coords(mws1gt_1,97,110,272).  visible(mws1gt_1,272). happensAt(walking(mws1gt_1),272). orientation(mws1gt_3,157,272). coords(mws1gt_3,87,132,272).  visible(mws1gt_3,272). happensAt(walking(mws1gt_3),272).
orientation(mws1gt_1,155,273). coords(mws1gt_1,97,110,273).  visible(mws1gt_1,273). happensAt(walking(mws1gt_1),273). orientation(mws1gt_3,157,273). coords(mws1gt_3,87,132,273).  visible(mws1gt_3,273). happensAt(walking(mws1gt_3),273).
orientation(mws1gt_1,155,274). coords(mws1gt_1,95,109,274).  visible(mws1gt_1,274). happensAt(walking(mws1gt_1),274). orientation(mws1gt_3,155,274). coords(mws1gt_3,87,132,274).  visible(mws1gt_3,274). happensAt(walking(mws1gt_3),274).
orientation(mws1gt_1,155,275). coords(mws1gt_1,95,109,275).  visible(mws1gt_1,275). happensAt(walking(mws1gt_1),275). orientation(mws1gt_3,155,275). coords(mws1gt_3,87,132,275).  visible(mws1gt_3,275). happensAt(walking(mws1gt_3),275).
orientation(mws1gt_1,155,276). coords(mws1gt_1,93,109,276).  visible(mws1gt_1,276). happensAt(walking(mws1gt_1),276). orientation(mws1gt_3,155,276). coords(mws1gt_3,87,132,276).  visible(mws1gt_3,276). happensAt(walking(mws1gt_3),276).
orientation(mws1gt_1,155,277). coords(mws1gt_1,93,109,277).  visible(mws1gt_1,277). happensAt(walking(mws1gt_1),277). orientation(mws1gt_3,155,277). coords(mws1gt_3,86,132,277).  visible(mws1gt_3,277). happensAt(walking(mws1gt_3),277).
orientation(mws1gt_1,155,278). coords(mws1gt_1,93,109,278).  visible(mws1gt_1,278). happensAt(walking(mws1gt_1),278). orientation(mws1gt_3,155,278). coords(mws1gt_3,86,131,278).  visible(mws1gt_3,278). happensAt(walking(mws1gt_3),278).
orientation(mws1gt_1,155,279). coords(mws1gt_1,93,109,279).  visible(mws1gt_1,279). happensAt(walking(mws1gt_1),279). orientation(mws1gt_3,155,279). coords(mws1gt_3,85,131,279).  visible(mws1gt_3,279). happensAt(walking(mws1gt_3),279).
orientation(mws1gt_1,155,280). coords(mws1gt_1,93,108,280).  visible(mws1gt_1,280). happensAt(walking(mws1gt_1),280). orientation(mws1gt_3,155,280). coords(mws1gt_3,85,131,280).  visible(mws1gt_3,280). happensAt(walking(mws1gt_3),280).
orientation(mws1gt_1,155,281). coords(mws1gt_1,93,108,281).  visible(mws1gt_1,281). happensAt(walking(mws1gt_1),281). orientation(mws1gt_3,155,281). coords(mws1gt_3,84,131,281).  visible(mws1gt_3,281). happensAt(walking(mws1gt_3),281).
orientation(mws1gt_1,155,282). coords(mws1gt_1,92,108,282).  visible(mws1gt_1,282). happensAt(walking(mws1gt_1),282). orientation(mws1gt_3,155,282). coords(mws1gt_3,83,131,282).  visible(mws1gt_3,282). happensAt(walking(mws1gt_3),282).
orientation(mws1gt_1,155,283). coords(mws1gt_1,92,108,283).  visible(mws1gt_1,283). happensAt(walking(mws1gt_1),283). orientation(mws1gt_3,155,283). coords(mws1gt_3,82,131,283).  visible(mws1gt_3,283). happensAt(walking(mws1gt_3),283).
orientation(mws1gt_1,155,284). coords(mws1gt_1,92,107,284).  visible(mws1gt_1,284). happensAt(walking(mws1gt_1),284). orientation(mws1gt_3,155,284). coords(mws1gt_3,82,131,284).  visible(mws1gt_3,284). happensAt(walking(mws1gt_3),284).
#show holdsAt/2.