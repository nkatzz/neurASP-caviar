
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

	orientation(wk2gt_1,125,264). coords(wk2gt_1,115,53,264).  visible(wk2gt_1,264). happensAt(inactive(wk2gt_1),264). orientation(wk2gt_2,141,264). coords(wk2gt_2,98,62,264).  visible(wk2gt_2,264). happensAt(inactive(wk2gt_2),264).
orientation(wk2gt_1,125,265). coords(wk2gt_1,115,53,265).  visible(wk2gt_1,265). happensAt(inactive(wk2gt_1),265). orientation(wk2gt_2,141,265). coords(wk2gt_2,98,62,265).  visible(wk2gt_2,265). happensAt(inactive(wk2gt_2),265).
orientation(wk2gt_1,125,266). coords(wk2gt_1,115,53,266).  visible(wk2gt_1,266). happensAt(inactive(wk2gt_1),266). orientation(wk2gt_2,141,266). coords(wk2gt_2,98,62,266).  visible(wk2gt_2,266). happensAt(inactive(wk2gt_2),266).
orientation(wk2gt_1,125,267). coords(wk2gt_1,115,53,267).  visible(wk2gt_1,267). happensAt(inactive(wk2gt_1),267). orientation(wk2gt_2,141,267). coords(wk2gt_2,98,62,267).  visible(wk2gt_2,267). happensAt(inactive(wk2gt_2),267).
orientation(wk2gt_1,125,268). coords(wk2gt_1,115,53,268).  visible(wk2gt_1,268). happensAt(inactive(wk2gt_1),268). orientation(wk2gt_2,141,268). coords(wk2gt_2,98,62,268).  visible(wk2gt_2,268). happensAt(inactive(wk2gt_2),268).
orientation(wk2gt_1,125,269). coords(wk2gt_1,115,53,269).  visible(wk2gt_1,269). happensAt(inactive(wk2gt_1),269). orientation(wk2gt_2,141,269). coords(wk2gt_2,98,62,269).  visible(wk2gt_2,269). happensAt(inactive(wk2gt_2),269).
orientation(wk2gt_1,125,270). coords(wk2gt_1,115,53,270).  visible(wk2gt_1,270). happensAt(inactive(wk2gt_1),270). orientation(wk2gt_2,141,270). coords(wk2gt_2,98,62,270).  visible(wk2gt_2,270). happensAt(inactive(wk2gt_2),270).
orientation(wk2gt_1,125,271). coords(wk2gt_1,115,53,271).  visible(wk2gt_1,271). happensAt(inactive(wk2gt_1),271). orientation(wk2gt_2,141,271). coords(wk2gt_2,98,62,271).  visible(wk2gt_2,271). happensAt(inactive(wk2gt_2),271).
orientation(wk2gt_1,125,272). coords(wk2gt_1,115,53,272).  visible(wk2gt_1,272). happensAt(inactive(wk2gt_1),272). orientation(wk2gt_2,141,272). coords(wk2gt_2,98,62,272).  visible(wk2gt_2,272). happensAt(inactive(wk2gt_2),272).
orientation(wk2gt_1,125,273). coords(wk2gt_1,115,53,273).  visible(wk2gt_1,273). happensAt(inactive(wk2gt_1),273). orientation(wk2gt_2,141,273). coords(wk2gt_2,98,62,273).  visible(wk2gt_2,273). happensAt(inactive(wk2gt_2),273).
orientation(wk2gt_1,125,274). coords(wk2gt_1,115,53,274).  visible(wk2gt_1,274). happensAt(inactive(wk2gt_1),274). orientation(wk2gt_2,141,274). coords(wk2gt_2,98,62,274).  visible(wk2gt_2,274). happensAt(inactive(wk2gt_2),274).
orientation(wk2gt_1,125,275). coords(wk2gt_1,115,53,275).  visible(wk2gt_1,275). happensAt(inactive(wk2gt_1),275). orientation(wk2gt_2,141,275). coords(wk2gt_2,98,62,275).  visible(wk2gt_2,275). happensAt(inactive(wk2gt_2),275).
orientation(wk2gt_1,125,276). coords(wk2gt_1,115,53,276).  visible(wk2gt_1,276). happensAt(inactive(wk2gt_1),276). orientation(wk2gt_2,141,276). coords(wk2gt_2,98,63,276).  visible(wk2gt_2,276). happensAt(inactive(wk2gt_2),276).
orientation(wk2gt_1,125,277). coords(wk2gt_1,115,53,277).  visible(wk2gt_1,277). happensAt(inactive(wk2gt_1),277). orientation(wk2gt_2,141,277). coords(wk2gt_2,98,63,277).  visible(wk2gt_2,277). happensAt(inactive(wk2gt_2),277).
orientation(wk2gt_1,125,278). coords(wk2gt_1,115,53,278).  visible(wk2gt_1,278). happensAt(inactive(wk2gt_1),278). orientation(wk2gt_2,141,278). coords(wk2gt_2,98,63,278).  visible(wk2gt_2,278). happensAt(inactive(wk2gt_2),278).
orientation(wk2gt_1,125,279). coords(wk2gt_1,115,53,279).  visible(wk2gt_1,279). happensAt(inactive(wk2gt_1),279). orientation(wk2gt_2,141,279). coords(wk2gt_2,98,63,279).  visible(wk2gt_2,279). happensAt(inactive(wk2gt_2),279).
orientation(wk2gt_1,125,280). coords(wk2gt_1,115,53,280).  visible(wk2gt_1,280). happensAt(inactive(wk2gt_1),280). orientation(wk2gt_2,141,280). coords(wk2gt_2,98,63,280).  visible(wk2gt_2,280). happensAt(inactive(wk2gt_2),280).
orientation(wk2gt_1,125,281). coords(wk2gt_1,115,53,281).  visible(wk2gt_1,281). happensAt(inactive(wk2gt_1),281). orientation(wk2gt_2,141,281). coords(wk2gt_2,98,63,281).  visible(wk2gt_2,281). happensAt(inactive(wk2gt_2),281).
orientation(wk2gt_1,125,282). coords(wk2gt_1,115,53,282).  visible(wk2gt_1,282). happensAt(inactive(wk2gt_1),282). orientation(wk2gt_2,141,282). coords(wk2gt_2,98,63,282).  visible(wk2gt_2,282). happensAt(inactive(wk2gt_2),282).
orientation(wk2gt_1,125,283). coords(wk2gt_1,115,53,283).  visible(wk2gt_1,283). happensAt(inactive(wk2gt_1),283). orientation(wk2gt_2,141,283). coords(wk2gt_2,98,63,283).  visible(wk2gt_2,283). happensAt(inactive(wk2gt_2),283).
orientation(wk2gt_1,125,284). coords(wk2gt_1,115,53,284).  visible(wk2gt_1,284). happensAt(inactive(wk2gt_1),284). orientation(wk2gt_2,141,284). coords(wk2gt_2,98,63,284).  visible(wk2gt_2,284). happensAt(inactive(wk2gt_2),284).
orientation(wk2gt_1,125,285). coords(wk2gt_1,115,53,285).  visible(wk2gt_1,285). happensAt(inactive(wk2gt_1),285). orientation(wk2gt_2,141,285). coords(wk2gt_2,98,63,285).  visible(wk2gt_2,285). happensAt(inactive(wk2gt_2),285).
orientation(wk2gt_1,125,286). coords(wk2gt_1,115,53,286).  visible(wk2gt_1,286). happensAt(inactive(wk2gt_1),286). orientation(wk2gt_2,141,286). coords(wk2gt_2,98,63,286).  visible(wk2gt_2,286). happensAt(inactive(wk2gt_2),286).
orientation(wk2gt_1,125,287). coords(wk2gt_1,115,53,287).  visible(wk2gt_1,287). happensAt(inactive(wk2gt_1),287). orientation(wk2gt_2,141,287). coords(wk2gt_2,98,63,287).  visible(wk2gt_2,287). happensAt(inactive(wk2gt_2),287).
orientation(wk2gt_1,125,288). coords(wk2gt_1,115,53,288).  visible(wk2gt_1,288). happensAt(inactive(wk2gt_1),288). orientation(wk2gt_2,141,288). coords(wk2gt_2,98,63,288).  visible(wk2gt_2,288). happensAt(inactive(wk2gt_2),288).
#show holdsAt/2.