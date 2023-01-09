
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

	orientation(mws1gt_1,155,262). coords(mws1gt_1,101,113,262).  visible(mws1gt_1,262). happensAt(walking(mws1gt_1),262). orientation(mws1gt_5,161,262). coords(mws1gt_5,41,150,262).  visible(mws1gt_5,262). happensAt(walking(mws1gt_5),262).
orientation(mws1gt_1,155,263). coords(mws1gt_1,101,113,263).  visible(mws1gt_1,263). happensAt(walking(mws1gt_1),263). orientation(mws1gt_5,161,263). coords(mws1gt_5,41,150,263).  visible(mws1gt_5,263). happensAt(walking(mws1gt_5),263).
orientation(mws1gt_1,155,264). coords(mws1gt_1,100,112,264).  visible(mws1gt_1,264). happensAt(walking(mws1gt_1),264). orientation(mws1gt_5,161,264). coords(mws1gt_5,41,151,264).  visible(mws1gt_5,264). happensAt(walking(mws1gt_5),264).
orientation(mws1gt_1,155,265). coords(mws1gt_1,100,111,265).  visible(mws1gt_1,265). happensAt(walking(mws1gt_1),265). orientation(mws1gt_5,161,265). coords(mws1gt_5,41,153,265).  visible(mws1gt_5,265). happensAt(walking(mws1gt_5),265).
orientation(mws1gt_1,155,266). coords(mws1gt_1,100,111,266).  visible(mws1gt_1,266). happensAt(walking(mws1gt_1),266). orientation(mws1gt_5,161,266). coords(mws1gt_5,41,153,266).  visible(mws1gt_5,266). happensAt(walking(mws1gt_5),266).
orientation(mws1gt_1,155,267). coords(mws1gt_1,100,111,267).  visible(mws1gt_1,267). happensAt(walking(mws1gt_1),267). orientation(mws1gt_5,161,267). coords(mws1gt_5,41,153,267).  visible(mws1gt_5,267). happensAt(walking(mws1gt_5),267).
orientation(mws1gt_1,155,268). coords(mws1gt_1,100,111,268).  visible(mws1gt_1,268). happensAt(walking(mws1gt_1),268). orientation(mws1gt_5,161,268). coords(mws1gt_5,41,153,268).  visible(mws1gt_5,268). happensAt(walking(mws1gt_5),268).
orientation(mws1gt_1,155,269). coords(mws1gt_1,99,111,269).  visible(mws1gt_1,269). happensAt(walking(mws1gt_1),269). orientation(mws1gt_5,161,269). coords(mws1gt_5,41,153,269).  visible(mws1gt_5,269). happensAt(walking(mws1gt_5),269).
orientation(mws1gt_1,155,270). coords(mws1gt_1,99,111,270).  visible(mws1gt_1,270). happensAt(walking(mws1gt_1),270). orientation(mws1gt_5,161,270). coords(mws1gt_5,40,153,270).  visible(mws1gt_5,270). happensAt(walking(mws1gt_5),270).
orientation(mws1gt_1,155,271). coords(mws1gt_1,98,111,271).  visible(mws1gt_1,271). happensAt(walking(mws1gt_1),271). orientation(mws1gt_5,161,271). coords(mws1gt_5,40,154,271).  visible(mws1gt_5,271). happensAt(walking(mws1gt_5),271).
orientation(mws1gt_1,155,272). coords(mws1gt_1,97,110,272).  visible(mws1gt_1,272). happensAt(walking(mws1gt_1),272). orientation(mws1gt_5,161,272). coords(mws1gt_5,40,154,272).  visible(mws1gt_5,272). happensAt(walking(mws1gt_5),272).
orientation(mws1gt_1,155,273). coords(mws1gt_1,97,110,273).  visible(mws1gt_1,273). happensAt(walking(mws1gt_1),273). orientation(mws1gt_5,161,273). coords(mws1gt_5,40,155,273).  visible(mws1gt_5,273). happensAt(walking(mws1gt_5),273).
orientation(mws1gt_1,155,274). coords(mws1gt_1,95,109,274).  visible(mws1gt_1,274). happensAt(walking(mws1gt_1),274). orientation(mws1gt_5,161,274). coords(mws1gt_5,40,156,274).  visible(mws1gt_5,274). happensAt(walking(mws1gt_5),274).
orientation(mws1gt_1,155,275). coords(mws1gt_1,95,109,275).  visible(mws1gt_1,275). happensAt(walking(mws1gt_1),275). orientation(mws1gt_5,161,275). coords(mws1gt_5,40,157,275).  visible(mws1gt_5,275). happensAt(walking(mws1gt_5),275).
orientation(mws1gt_1,155,276). coords(mws1gt_1,93,109,276).  visible(mws1gt_1,276). happensAt(walking(mws1gt_1),276). orientation(mws1gt_5,161,276). coords(mws1gt_5,39,158,276).  visible(mws1gt_5,276). happensAt(walking(mws1gt_5),276).
orientation(mws1gt_1,155,277). coords(mws1gt_1,93,109,277).  visible(mws1gt_1,277). happensAt(walking(mws1gt_1),277). orientation(mws1gt_5,161,277). coords(mws1gt_5,39,159,277).  visible(mws1gt_5,277). happensAt(walking(mws1gt_5),277).
orientation(mws1gt_1,155,278). coords(mws1gt_1,93,109,278).  visible(mws1gt_1,278). happensAt(walking(mws1gt_1),278). orientation(mws1gt_5,161,278). coords(mws1gt_5,38,160,278).  visible(mws1gt_5,278). happensAt(walking(mws1gt_5),278).
orientation(mws1gt_1,155,279). coords(mws1gt_1,93,109,279).  visible(mws1gt_1,279). happensAt(walking(mws1gt_1),279). orientation(mws1gt_5,161,279). coords(mws1gt_5,38,162,279).  visible(mws1gt_5,279). happensAt(walking(mws1gt_5),279).
orientation(mws1gt_1,155,280). coords(mws1gt_1,93,108,280).  visible(mws1gt_1,280). happensAt(walking(mws1gt_1),280). orientation(mws1gt_5,161,280). coords(mws1gt_5,38,162,280).  visible(mws1gt_5,280). happensAt(walking(mws1gt_5),280).
orientation(mws1gt_1,155,281). coords(mws1gt_1,93,108,281).  visible(mws1gt_1,281). happensAt(walking(mws1gt_1),281). orientation(mws1gt_5,161,281). coords(mws1gt_5,38,163,281).  visible(mws1gt_5,281). happensAt(walking(mws1gt_5),281).
orientation(mws1gt_1,155,282). coords(mws1gt_1,92,108,282).  visible(mws1gt_1,282). happensAt(walking(mws1gt_1),282). orientation(mws1gt_5,161,282). coords(mws1gt_5,38,164,282).  visible(mws1gt_5,282). happensAt(walking(mws1gt_5),282).
orientation(mws1gt_1,155,283). coords(mws1gt_1,92,108,283).  visible(mws1gt_1,283). happensAt(walking(mws1gt_1),283). orientation(mws1gt_5,161,283). coords(mws1gt_5,38,164,283).  visible(mws1gt_5,283). happensAt(walking(mws1gt_5),283).
orientation(mws1gt_1,155,284). coords(mws1gt_1,92,107,284).  visible(mws1gt_1,284). happensAt(walking(mws1gt_1),284). orientation(mws1gt_5,161,284). coords(mws1gt_5,37,164,284).  visible(mws1gt_5,284). happensAt(walking(mws1gt_5),284).
orientation(mws1gt_1,155,285). coords(mws1gt_1,91,107,285).  visible(mws1gt_1,285). happensAt(walking(mws1gt_1),285). orientation(mws1gt_5,161,285). coords(mws1gt_5,37,165,285).  visible(mws1gt_5,285). happensAt(walking(mws1gt_5),285).
orientation(mws1gt_1,155,286). coords(mws1gt_1,91,106,286).  visible(mws1gt_1,286). happensAt(walking(mws1gt_1),286). orientation(mws1gt_5,161,286). coords(mws1gt_5,37,165,286).  visible(mws1gt_5,286). happensAt(walking(mws1gt_5),286).
#show holdsAt/2.