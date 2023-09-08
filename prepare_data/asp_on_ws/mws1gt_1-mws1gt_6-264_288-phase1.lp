
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

	orientation(mws1gt_1,155,264). coords(mws1gt_1,100,112,264).  visible(mws1gt_1,264). happensAt(walking(mws1gt_1),264). orientation(mws1gt_6,146,264). coords(mws1gt_6,42,129,264).  visible(mws1gt_6,264). happensAt(walking(mws1gt_6),264).
orientation(mws1gt_1,155,265). coords(mws1gt_1,100,111,265).  visible(mws1gt_1,265). happensAt(walking(mws1gt_1),265). orientation(mws1gt_6,146,265). coords(mws1gt_6,42,129,265).  visible(mws1gt_6,265). happensAt(walking(mws1gt_6),265).
orientation(mws1gt_1,155,266). coords(mws1gt_1,100,111,266).  visible(mws1gt_1,266). happensAt(walking(mws1gt_1),266). orientation(mws1gt_6,146,266). coords(mws1gt_6,42,130,266).  visible(mws1gt_6,266). happensAt(walking(mws1gt_6),266).
orientation(mws1gt_1,155,267). coords(mws1gt_1,100,111,267).  visible(mws1gt_1,267). happensAt(walking(mws1gt_1),267). orientation(mws1gt_6,146,267). coords(mws1gt_6,42,130,267).  visible(mws1gt_6,267). happensAt(walking(mws1gt_6),267).
orientation(mws1gt_1,155,268). coords(mws1gt_1,100,111,268).  visible(mws1gt_1,268). happensAt(walking(mws1gt_1),268). orientation(mws1gt_6,146,268). coords(mws1gt_6,42,130,268).  visible(mws1gt_6,268). happensAt(walking(mws1gt_6),268).
orientation(mws1gt_1,155,269). coords(mws1gt_1,99,111,269).  visible(mws1gt_1,269). happensAt(walking(mws1gt_1),269). orientation(mws1gt_6,146,269). coords(mws1gt_6,42,131,269).  visible(mws1gt_6,269). happensAt(walking(mws1gt_6),269).
orientation(mws1gt_1,155,270). coords(mws1gt_1,99,111,270).  visible(mws1gt_1,270). happensAt(walking(mws1gt_1),270). orientation(mws1gt_6,146,270). coords(mws1gt_6,42,132,270).  visible(mws1gt_6,270). happensAt(walking(mws1gt_6),270).
orientation(mws1gt_1,155,271). coords(mws1gt_1,98,111,271).  visible(mws1gt_1,271). happensAt(walking(mws1gt_1),271). orientation(mws1gt_6,146,271). coords(mws1gt_6,41,133,271).  visible(mws1gt_6,271). happensAt(walking(mws1gt_6),271).
orientation(mws1gt_1,155,272). coords(mws1gt_1,97,110,272).  visible(mws1gt_1,272). happensAt(walking(mws1gt_1),272). orientation(mws1gt_6,146,272). coords(mws1gt_6,41,133,272).  visible(mws1gt_6,272). happensAt(walking(mws1gt_6),272).
orientation(mws1gt_1,155,273). coords(mws1gt_1,97,110,273).  visible(mws1gt_1,273). happensAt(walking(mws1gt_1),273). orientation(mws1gt_6,146,273). coords(mws1gt_6,40,134,273).  visible(mws1gt_6,273). happensAt(walking(mws1gt_6),273).
orientation(mws1gt_1,155,274). coords(mws1gt_1,95,109,274).  visible(mws1gt_1,274). happensAt(walking(mws1gt_1),274). orientation(mws1gt_6,146,274). coords(mws1gt_6,40,134,274).  visible(mws1gt_6,274). happensAt(walking(mws1gt_6),274).
orientation(mws1gt_1,155,275). coords(mws1gt_1,95,109,275).  visible(mws1gt_1,275). happensAt(walking(mws1gt_1),275). orientation(mws1gt_6,146,275). coords(mws1gt_6,40,135,275).  visible(mws1gt_6,275). happensAt(walking(mws1gt_6),275).
orientation(mws1gt_1,155,276). coords(mws1gt_1,93,109,276).  visible(mws1gt_1,276). happensAt(walking(mws1gt_1),276). orientation(mws1gt_6,146,276). coords(mws1gt_6,40,136,276).  visible(mws1gt_6,276). happensAt(walking(mws1gt_6),276).
orientation(mws1gt_1,155,277). coords(mws1gt_1,93,109,277).  visible(mws1gt_1,277). happensAt(walking(mws1gt_1),277). orientation(mws1gt_6,146,277). coords(mws1gt_6,40,137,277).  visible(mws1gt_6,277). happensAt(walking(mws1gt_6),277).
orientation(mws1gt_1,155,278). coords(mws1gt_1,93,109,278).  visible(mws1gt_1,278). happensAt(walking(mws1gt_1),278). orientation(mws1gt_6,146,278). coords(mws1gt_6,40,138,278).  visible(mws1gt_6,278). happensAt(walking(mws1gt_6),278).
orientation(mws1gt_1,155,279). coords(mws1gt_1,93,109,279).  visible(mws1gt_1,279). happensAt(walking(mws1gt_1),279). orientation(mws1gt_6,146,279). coords(mws1gt_6,40,139,279).  visible(mws1gt_6,279). happensAt(walking(mws1gt_6),279).
orientation(mws1gt_1,155,280). coords(mws1gt_1,93,108,280).  visible(mws1gt_1,280). happensAt(walking(mws1gt_1),280). orientation(mws1gt_6,146,280). coords(mws1gt_6,40,140,280).  visible(mws1gt_6,280). happensAt(walking(mws1gt_6),280).
orientation(mws1gt_1,155,281). coords(mws1gt_1,93,108,281).  visible(mws1gt_1,281). happensAt(walking(mws1gt_1),281). orientation(mws1gt_6,146,281). coords(mws1gt_6,40,140,281).  visible(mws1gt_6,281). happensAt(walking(mws1gt_6),281).
orientation(mws1gt_1,155,282). coords(mws1gt_1,92,108,282).  visible(mws1gt_1,282). happensAt(walking(mws1gt_1),282). orientation(mws1gt_6,146,282). coords(mws1gt_6,39,140,282).  visible(mws1gt_6,282). happensAt(walking(mws1gt_6),282).
orientation(mws1gt_1,155,283). coords(mws1gt_1,92,108,283).  visible(mws1gt_1,283). happensAt(walking(mws1gt_1),283). orientation(mws1gt_6,146,283). coords(mws1gt_6,38,142,283).  visible(mws1gt_6,283). happensAt(walking(mws1gt_6),283).
orientation(mws1gt_1,155,284). coords(mws1gt_1,92,107,284).  visible(mws1gt_1,284). happensAt(walking(mws1gt_1),284). orientation(mws1gt_6,146,284). coords(mws1gt_6,38,143,284).  visible(mws1gt_6,284). happensAt(walking(mws1gt_6),284).
orientation(mws1gt_1,155,285). coords(mws1gt_1,91,107,285).  visible(mws1gt_1,285). happensAt(walking(mws1gt_1),285). orientation(mws1gt_6,146,285). coords(mws1gt_6,38,144,285).  visible(mws1gt_6,285). happensAt(walking(mws1gt_6),285).
orientation(mws1gt_1,155,286). coords(mws1gt_1,91,106,286).  visible(mws1gt_1,286). happensAt(walking(mws1gt_1),286). orientation(mws1gt_6,146,286). coords(mws1gt_6,38,145,286).  visible(mws1gt_6,286). happensAt(walking(mws1gt_6),286).
orientation(mws1gt_1,155,287). coords(mws1gt_1,89,106,287).  visible(mws1gt_1,287). happensAt(walking(mws1gt_1),287). orientation(mws1gt_6,146,287). coords(mws1gt_6,38,146,287).  visible(mws1gt_6,287). happensAt(walking(mws1gt_6),287).
orientation(mws1gt_1,155,288). coords(mws1gt_1,88,105,288).  visible(mws1gt_1,288). happensAt(walking(mws1gt_1),288). orientation(mws1gt_6,146,288). coords(mws1gt_6,38,146,288).  visible(mws1gt_6,288). happensAt(walking(mws1gt_6),288).
#show holdsAt/2.