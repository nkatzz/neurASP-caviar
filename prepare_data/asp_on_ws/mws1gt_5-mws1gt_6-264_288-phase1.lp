
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

	orientation(mws1gt_5,161,264). coords(mws1gt_5,41,151,264).  visible(mws1gt_5,264). happensAt(walking(mws1gt_5),264). orientation(mws1gt_6,146,264). coords(mws1gt_6,42,129,264).  visible(mws1gt_6,264). happensAt(walking(mws1gt_6),264).
orientation(mws1gt_5,161,265). coords(mws1gt_5,41,153,265).  visible(mws1gt_5,265). happensAt(walking(mws1gt_5),265). orientation(mws1gt_6,146,265). coords(mws1gt_6,42,129,265).  visible(mws1gt_6,265). happensAt(walking(mws1gt_6),265).
orientation(mws1gt_5,161,266). coords(mws1gt_5,41,153,266).  visible(mws1gt_5,266). happensAt(walking(mws1gt_5),266). orientation(mws1gt_6,146,266). coords(mws1gt_6,42,130,266).  visible(mws1gt_6,266). happensAt(walking(mws1gt_6),266).
orientation(mws1gt_5,161,267). coords(mws1gt_5,41,153,267).  visible(mws1gt_5,267). happensAt(walking(mws1gt_5),267). orientation(mws1gt_6,146,267). coords(mws1gt_6,42,130,267).  visible(mws1gt_6,267). happensAt(walking(mws1gt_6),267).
orientation(mws1gt_5,161,268). coords(mws1gt_5,41,153,268).  visible(mws1gt_5,268). happensAt(walking(mws1gt_5),268). orientation(mws1gt_6,146,268). coords(mws1gt_6,42,130,268).  visible(mws1gt_6,268). happensAt(walking(mws1gt_6),268).
orientation(mws1gt_5,161,269). coords(mws1gt_5,41,153,269).  visible(mws1gt_5,269). happensAt(walking(mws1gt_5),269). orientation(mws1gt_6,146,269). coords(mws1gt_6,42,131,269).  visible(mws1gt_6,269). happensAt(walking(mws1gt_6),269).
orientation(mws1gt_5,161,270). coords(mws1gt_5,40,153,270).  visible(mws1gt_5,270). happensAt(walking(mws1gt_5),270). orientation(mws1gt_6,146,270). coords(mws1gt_6,42,132,270).  visible(mws1gt_6,270). happensAt(walking(mws1gt_6),270).
orientation(mws1gt_5,161,271). coords(mws1gt_5,40,154,271).  visible(mws1gt_5,271). happensAt(walking(mws1gt_5),271). orientation(mws1gt_6,146,271). coords(mws1gt_6,41,133,271).  visible(mws1gt_6,271). happensAt(walking(mws1gt_6),271).
orientation(mws1gt_5,161,272). coords(mws1gt_5,40,154,272).  visible(mws1gt_5,272). happensAt(walking(mws1gt_5),272). orientation(mws1gt_6,146,272). coords(mws1gt_6,41,133,272).  visible(mws1gt_6,272). happensAt(walking(mws1gt_6),272).
orientation(mws1gt_5,161,273). coords(mws1gt_5,40,155,273).  visible(mws1gt_5,273). happensAt(walking(mws1gt_5),273). orientation(mws1gt_6,146,273). coords(mws1gt_6,40,134,273).  visible(mws1gt_6,273). happensAt(walking(mws1gt_6),273).
orientation(mws1gt_5,161,274). coords(mws1gt_5,40,156,274).  visible(mws1gt_5,274). happensAt(walking(mws1gt_5),274). orientation(mws1gt_6,146,274). coords(mws1gt_6,40,134,274).  visible(mws1gt_6,274). happensAt(walking(mws1gt_6),274).
orientation(mws1gt_5,161,275). coords(mws1gt_5,40,157,275).  visible(mws1gt_5,275). happensAt(walking(mws1gt_5),275). orientation(mws1gt_6,146,275). coords(mws1gt_6,40,135,275).  visible(mws1gt_6,275). happensAt(walking(mws1gt_6),275).
orientation(mws1gt_5,161,276). coords(mws1gt_5,39,158,276).  visible(mws1gt_5,276). happensAt(walking(mws1gt_5),276). orientation(mws1gt_6,146,276). coords(mws1gt_6,40,136,276).  visible(mws1gt_6,276). happensAt(walking(mws1gt_6),276).
orientation(mws1gt_5,161,277). coords(mws1gt_5,39,159,277).  visible(mws1gt_5,277). happensAt(walking(mws1gt_5),277). orientation(mws1gt_6,146,277). coords(mws1gt_6,40,137,277).  visible(mws1gt_6,277). happensAt(walking(mws1gt_6),277).
orientation(mws1gt_5,161,278). coords(mws1gt_5,38,160,278).  visible(mws1gt_5,278). happensAt(walking(mws1gt_5),278). orientation(mws1gt_6,146,278). coords(mws1gt_6,40,138,278).  visible(mws1gt_6,278). happensAt(walking(mws1gt_6),278).
orientation(mws1gt_5,161,279). coords(mws1gt_5,38,162,279).  visible(mws1gt_5,279). happensAt(walking(mws1gt_5),279). orientation(mws1gt_6,146,279). coords(mws1gt_6,40,139,279).  visible(mws1gt_6,279). happensAt(walking(mws1gt_6),279).
orientation(mws1gt_5,161,280). coords(mws1gt_5,38,162,280).  visible(mws1gt_5,280). happensAt(walking(mws1gt_5),280). orientation(mws1gt_6,146,280). coords(mws1gt_6,40,140,280).  visible(mws1gt_6,280). happensAt(walking(mws1gt_6),280).
orientation(mws1gt_5,161,281). coords(mws1gt_5,38,163,281).  visible(mws1gt_5,281). happensAt(walking(mws1gt_5),281). orientation(mws1gt_6,146,281). coords(mws1gt_6,40,140,281).  visible(mws1gt_6,281). happensAt(walking(mws1gt_6),281).
orientation(mws1gt_5,161,282). coords(mws1gt_5,38,164,282).  visible(mws1gt_5,282). happensAt(walking(mws1gt_5),282). orientation(mws1gt_6,146,282). coords(mws1gt_6,39,140,282).  visible(mws1gt_6,282). happensAt(walking(mws1gt_6),282).
orientation(mws1gt_5,161,283). coords(mws1gt_5,38,164,283).  visible(mws1gt_5,283). happensAt(walking(mws1gt_5),283). orientation(mws1gt_6,146,283). coords(mws1gt_6,38,142,283).  visible(mws1gt_6,283). happensAt(walking(mws1gt_6),283).
orientation(mws1gt_5,161,284). coords(mws1gt_5,37,164,284).  visible(mws1gt_5,284). happensAt(walking(mws1gt_5),284). orientation(mws1gt_6,146,284). coords(mws1gt_6,38,143,284).  visible(mws1gt_6,284). happensAt(walking(mws1gt_6),284).
orientation(mws1gt_5,161,285). coords(mws1gt_5,37,165,285).  visible(mws1gt_5,285). happensAt(walking(mws1gt_5),285). orientation(mws1gt_6,146,285). coords(mws1gt_6,38,144,285).  visible(mws1gt_6,285). happensAt(walking(mws1gt_6),285).
orientation(mws1gt_5,161,286). coords(mws1gt_5,37,165,286).  visible(mws1gt_5,286). happensAt(walking(mws1gt_5),286). orientation(mws1gt_6,146,286). coords(mws1gt_6,38,145,286).  visible(mws1gt_6,286). happensAt(walking(mws1gt_6),286).
orientation(mws1gt_5,161,287). coords(mws1gt_5,37,166,287).  visible(mws1gt_5,287). happensAt(walking(mws1gt_5),287). orientation(mws1gt_6,146,287). coords(mws1gt_6,38,146,287).  visible(mws1gt_6,287). happensAt(walking(mws1gt_6),287).
orientation(mws1gt_5,161,288). coords(mws1gt_5,37,166,288).  visible(mws1gt_5,288). happensAt(walking(mws1gt_5),288). orientation(mws1gt_6,146,288). coords(mws1gt_6,38,146,288).  visible(mws1gt_6,288). happensAt(walking(mws1gt_6),288).
#show holdsAt/2.