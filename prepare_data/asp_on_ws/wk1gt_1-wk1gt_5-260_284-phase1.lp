
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

	orientation(wk1gt_1,11,260). coords(wk1gt_1,224,264,260).  visible(wk1gt_1,260). happensAt(walking(wk1gt_1),260). orientation(wk1gt_5,162,260). coords(wk1gt_5,50,166,260).  visible(wk1gt_5,260). happensAt(inactive(wk1gt_5),260).
orientation(wk1gt_1,17,261). coords(wk1gt_1,224,260,261).  visible(wk1gt_1,261). happensAt(walking(wk1gt_1),261). orientation(wk1gt_5,162,261). coords(wk1gt_5,50,166,261).  visible(wk1gt_5,261). happensAt(inactive(wk1gt_5),261).
orientation(wk1gt_1,22,262). coords(wk1gt_1,224,256,262).  visible(wk1gt_1,262). happensAt(walking(wk1gt_1),262). orientation(wk1gt_5,162,262). coords(wk1gt_5,50,166,262).  visible(wk1gt_5,262). happensAt(inactive(wk1gt_5),262).
orientation(wk1gt_1,22,263). coords(wk1gt_1,223,254,263).  visible(wk1gt_1,263). happensAt(walking(wk1gt_1),263). orientation(wk1gt_5,162,263). coords(wk1gt_5,50,166,263).  visible(wk1gt_5,263). happensAt(inactive(wk1gt_5),263).
orientation(wk1gt_1,22,264). coords(wk1gt_1,223,252,264).  visible(wk1gt_1,264). happensAt(walking(wk1gt_1),264). orientation(wk1gt_5,162,264). coords(wk1gt_5,50,166,264).  visible(wk1gt_5,264). happensAt(inactive(wk1gt_5),264).
orientation(wk1gt_1,22,265). coords(wk1gt_1,222,251,265).  visible(wk1gt_1,265). happensAt(walking(wk1gt_1),265). orientation(wk1gt_5,162,265). coords(wk1gt_5,50,166,265).  visible(wk1gt_5,265). happensAt(inactive(wk1gt_5),265).
orientation(wk1gt_1,22,266). coords(wk1gt_1,222,251,266).  visible(wk1gt_1,266). happensAt(walking(wk1gt_1),266). orientation(wk1gt_5,162,266). coords(wk1gt_5,50,166,266).  visible(wk1gt_5,266). happensAt(inactive(wk1gt_5),266).
orientation(wk1gt_1,8,267). coords(wk1gt_1,221,250,267).  visible(wk1gt_1,267). happensAt(walking(wk1gt_1),267). orientation(wk1gt_5,162,267). coords(wk1gt_5,50,166,267).  visible(wk1gt_5,267). happensAt(inactive(wk1gt_5),267).
orientation(wk1gt_1,8,268). coords(wk1gt_1,218,249,268).  visible(wk1gt_1,268). happensAt(walking(wk1gt_1),268). orientation(wk1gt_5,162,268). coords(wk1gt_5,50,166,268).  visible(wk1gt_5,268). happensAt(inactive(wk1gt_5),268).
orientation(wk1gt_1,8,269). coords(wk1gt_1,218,247,269).  visible(wk1gt_1,269). happensAt(walking(wk1gt_1),269). orientation(wk1gt_5,162,269). coords(wk1gt_5,50,166,269).  visible(wk1gt_5,269). happensAt(inactive(wk1gt_5),269).
orientation(wk1gt_1,3,270). coords(wk1gt_1,216,245,270).  visible(wk1gt_1,270). happensAt(walking(wk1gt_1),270). orientation(wk1gt_5,162,270). coords(wk1gt_5,50,166,270).  visible(wk1gt_5,270). happensAt(inactive(wk1gt_5),270).
orientation(wk1gt_1,3,271). coords(wk1gt_1,216,242,271).  visible(wk1gt_1,271). happensAt(walking(wk1gt_1),271). orientation(wk1gt_5,162,271). coords(wk1gt_5,50,166,271).  visible(wk1gt_5,271). happensAt(inactive(wk1gt_5),271).
orientation(wk1gt_1,3,272). coords(wk1gt_1,213,241,272).  visible(wk1gt_1,272). happensAt(walking(wk1gt_1),272). orientation(wk1gt_5,162,272). coords(wk1gt_5,50,166,272).  visible(wk1gt_5,272). happensAt(active(wk1gt_5),272).
orientation(wk1gt_1,3,273). coords(wk1gt_1,212,239,273).  visible(wk1gt_1,273). happensAt(walking(wk1gt_1),273). orientation(wk1gt_5,162,273). coords(wk1gt_5,50,166,273).  visible(wk1gt_5,273). happensAt(active(wk1gt_5),273).
orientation(wk1gt_1,3,274). coords(wk1gt_1,212,236,274).  visible(wk1gt_1,274). happensAt(walking(wk1gt_1),274). orientation(wk1gt_5,162,274). coords(wk1gt_5,50,166,274).  visible(wk1gt_5,274). happensAt(active(wk1gt_5),274).
orientation(wk1gt_1,3,275). coords(wk1gt_1,211,232,275).  visible(wk1gt_1,275). happensAt(walking(wk1gt_1),275). orientation(wk1gt_5,162,275). coords(wk1gt_5,50,166,275).  visible(wk1gt_5,275). happensAt(active(wk1gt_5),275).
orientation(wk1gt_1,9,276). coords(wk1gt_1,211,230,276).  visible(wk1gt_1,276). happensAt(walking(wk1gt_1),276). orientation(wk1gt_5,162,276). coords(wk1gt_5,50,166,276).  visible(wk1gt_5,276). happensAt(active(wk1gt_5),276).
orientation(wk1gt_1,9,277). coords(wk1gt_1,210,229,277).  visible(wk1gt_1,277). happensAt(walking(wk1gt_1),277). orientation(wk1gt_5,162,277). coords(wk1gt_5,50,166,277).  visible(wk1gt_5,277). happensAt(active(wk1gt_5),277).
orientation(wk1gt_1,174,278). coords(wk1gt_1,210,229,278).  visible(wk1gt_1,278). happensAt(walking(wk1gt_1),278). orientation(wk1gt_5,162,278). coords(wk1gt_5,50,166,278).  visible(wk1gt_5,278). happensAt(active(wk1gt_5),278).
orientation(wk1gt_1,167,279). coords(wk1gt_1,210,229,279).  visible(wk1gt_1,279). happensAt(walking(wk1gt_1),279). orientation(wk1gt_5,162,279). coords(wk1gt_5,50,166,279).  visible(wk1gt_5,279). happensAt(active(wk1gt_5),279).
orientation(wk1gt_1,167,280). coords(wk1gt_1,209,228,280).  visible(wk1gt_1,280). happensAt(walking(wk1gt_1),280). orientation(wk1gt_5,162,280). coords(wk1gt_5,50,166,280).  visible(wk1gt_5,280). happensAt(active(wk1gt_5),280).
orientation(wk1gt_1,155,281). coords(wk1gt_1,207,227,281).  visible(wk1gt_1,281). happensAt(walking(wk1gt_1),281). orientation(wk1gt_5,162,281). coords(wk1gt_5,50,166,281).  visible(wk1gt_5,281). happensAt(active(wk1gt_5),281).
orientation(wk1gt_1,155,282). coords(wk1gt_1,206,224,282).  visible(wk1gt_1,282). happensAt(walking(wk1gt_1),282). orientation(wk1gt_5,162,282). coords(wk1gt_5,50,166,282).  visible(wk1gt_5,282). happensAt(active(wk1gt_5),282).
orientation(wk1gt_1,155,283). coords(wk1gt_1,203,222,283).  visible(wk1gt_1,283). happensAt(walking(wk1gt_1),283). orientation(wk1gt_5,162,283). coords(wk1gt_5,50,166,283).  visible(wk1gt_5,283). happensAt(active(wk1gt_5),283).
orientation(wk1gt_1,155,284). coords(wk1gt_1,203,218,284).  visible(wk1gt_1,284). happensAt(walking(wk1gt_1),284). orientation(wk1gt_5,162,284). coords(wk1gt_5,50,166,284).  visible(wk1gt_5,284). happensAt(active(wk1gt_5),284).
#show holdsAt/2.