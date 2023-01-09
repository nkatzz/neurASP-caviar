
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

	orientation(lb1gt_1,151,264). coords(lb1gt_1,153,160,264).  visible(lb1gt_1,264). happensAt(active(lb1gt_1),264). orientation(lb1gt_2,149,264). coords(lb1gt_2,172,134,264).  visible(lb1gt_2,264). happensAt(walking(lb1gt_2),264).
orientation(lb1gt_1,151,265). coords(lb1gt_1,153,160,265).  visible(lb1gt_1,265). happensAt(active(lb1gt_1),265). orientation(lb1gt_2,149,265). coords(lb1gt_2,172,134,265).  visible(lb1gt_2,265). happensAt(walking(lb1gt_2),265).
orientation(lb1gt_1,151,266). coords(lb1gt_1,153,160,266).  visible(lb1gt_1,266). happensAt(active(lb1gt_1),266). orientation(lb1gt_2,149,266). coords(lb1gt_2,173,134,266).  visible(lb1gt_2,266). happensAt(walking(lb1gt_2),266).
orientation(lb1gt_1,151,267). coords(lb1gt_1,153,160,267).  visible(lb1gt_1,267). happensAt(active(lb1gt_1),267). orientation(lb1gt_2,149,267). coords(lb1gt_2,175,135,267).  visible(lb1gt_2,267). happensAt(walking(lb1gt_2),267).
orientation(lb1gt_1,151,268). coords(lb1gt_1,153,160,268).  visible(lb1gt_1,268). happensAt(active(lb1gt_1),268). orientation(lb1gt_2,149,268). coords(lb1gt_2,175,135,268).  visible(lb1gt_2,268). happensAt(walking(lb1gt_2),268).
orientation(lb1gt_1,151,269). coords(lb1gt_1,154,160,269).  visible(lb1gt_1,269). happensAt(active(lb1gt_1),269). orientation(lb1gt_2,149,269). coords(lb1gt_2,176,135,269).  visible(lb1gt_2,269). happensAt(walking(lb1gt_2),269).
orientation(lb1gt_1,151,270). coords(lb1gt_1,154,160,270).  visible(lb1gt_1,270). happensAt(active(lb1gt_1),270). orientation(lb1gt_2,149,270). coords(lb1gt_2,176,135,270).  visible(lb1gt_2,270). happensAt(walking(lb1gt_2),270).
orientation(lb1gt_1,151,271). coords(lb1gt_1,154,159,271).  visible(lb1gt_1,271). happensAt(active(lb1gt_1),271). orientation(lb1gt_2,149,271). coords(lb1gt_2,177,135,271).  visible(lb1gt_2,271). happensAt(walking(lb1gt_2),271).
orientation(lb1gt_1,151,272). coords(lb1gt_1,154,159,272).  visible(lb1gt_1,272). happensAt(active(lb1gt_1),272). orientation(lb1gt_2,146,272). coords(lb1gt_2,178,136,272).  visible(lb1gt_2,272). happensAt(walking(lb1gt_2),272).
orientation(lb1gt_1,151,273). coords(lb1gt_1,154,159,273).  visible(lb1gt_1,273). happensAt(active(lb1gt_1),273). orientation(lb1gt_2,146,273). coords(lb1gt_2,178,136,273).  visible(lb1gt_2,273). happensAt(walking(lb1gt_2),273).
orientation(lb1gt_1,151,274). coords(lb1gt_1,154,159,274).  visible(lb1gt_1,274). happensAt(active(lb1gt_1),274). orientation(lb1gt_2,146,274). coords(lb1gt_2,178,137,274).  visible(lb1gt_2,274). happensAt(walking(lb1gt_2),274).
orientation(lb1gt_1,151,275). coords(lb1gt_1,154,159,275).  visible(lb1gt_1,275). happensAt(active(lb1gt_1),275). orientation(lb1gt_2,146,275). coords(lb1gt_2,179,138,275).  visible(lb1gt_2,275). happensAt(walking(lb1gt_2),275).
orientation(lb1gt_1,151,276). coords(lb1gt_1,154,159,276).  visible(lb1gt_1,276). happensAt(active(lb1gt_1),276). orientation(lb1gt_2,146,276). coords(lb1gt_2,180,140,276).  visible(lb1gt_2,276). happensAt(walking(lb1gt_2),276).
orientation(lb1gt_1,151,277). coords(lb1gt_1,154,159,277).  visible(lb1gt_1,277). happensAt(active(lb1gt_1),277). orientation(lb1gt_2,146,277). coords(lb1gt_2,180,142,277).  visible(lb1gt_2,277). happensAt(walking(lb1gt_2),277).
orientation(lb1gt_1,151,278). coords(lb1gt_1,154,159,278).  visible(lb1gt_1,278). happensAt(active(lb1gt_1),278). orientation(lb1gt_2,146,278). coords(lb1gt_2,180,144,278).  visible(lb1gt_2,278). happensAt(walking(lb1gt_2),278).
orientation(lb1gt_1,151,279). coords(lb1gt_1,154,159,279).  visible(lb1gt_1,279). happensAt(active(lb1gt_1),279). orientation(lb1gt_2,146,279). coords(lb1gt_2,181,146,279).  visible(lb1gt_2,279). happensAt(walking(lb1gt_2),279).
orientation(lb1gt_1,151,280). coords(lb1gt_1,154,159,280).  visible(lb1gt_1,280). happensAt(active(lb1gt_1),280). orientation(lb1gt_2,146,280). coords(lb1gt_2,183,148,280).  visible(lb1gt_2,280). happensAt(walking(lb1gt_2),280).
orientation(lb1gt_1,151,281). coords(lb1gt_1,154,159,281).  visible(lb1gt_1,281). happensAt(active(lb1gt_1),281). orientation(lb1gt_2,146,281). coords(lb1gt_2,184,148,281).  visible(lb1gt_2,281). happensAt(walking(lb1gt_2),281).
orientation(lb1gt_1,151,282). coords(lb1gt_1,154,159,282).  visible(lb1gt_1,282). happensAt(active(lb1gt_1),282). orientation(lb1gt_2,146,282). coords(lb1gt_2,185,149,282).  visible(lb1gt_2,282). happensAt(walking(lb1gt_2),282).
orientation(lb1gt_1,151,283). coords(lb1gt_1,154,159,283).  visible(lb1gt_1,283). happensAt(active(lb1gt_1),283). orientation(lb1gt_2,146,283). coords(lb1gt_2,185,150,283).  visible(lb1gt_2,283). happensAt(walking(lb1gt_2),283).
orientation(lb1gt_1,151,284). coords(lb1gt_1,155,159,284).  visible(lb1gt_1,284). happensAt(active(lb1gt_1),284). orientation(lb1gt_2,146,284). coords(lb1gt_2,185,150,284).  visible(lb1gt_2,284). happensAt(walking(lb1gt_2),284).
orientation(lb1gt_1,151,285). coords(lb1gt_1,155,159,285).  visible(lb1gt_1,285). happensAt(active(lb1gt_1),285). orientation(lb1gt_2,146,285). coords(lb1gt_2,186,151,285).  visible(lb1gt_2,285). happensAt(walking(lb1gt_2),285).
orientation(lb1gt_1,151,286). coords(lb1gt_1,155,158,286).  visible(lb1gt_1,286). happensAt(active(lb1gt_1),286). orientation(lb1gt_2,146,286). coords(lb1gt_2,186,151,286).  visible(lb1gt_2,286). happensAt(walking(lb1gt_2),286).
orientation(lb1gt_1,151,287). coords(lb1gt_1,155,158,287).  visible(lb1gt_1,287). happensAt(active(lb1gt_1),287). orientation(lb1gt_2,146,287). coords(lb1gt_2,186,152,287).  visible(lb1gt_2,287). happensAt(walking(lb1gt_2),287).
orientation(lb1gt_1,151,288). coords(lb1gt_1,155,158,288).  visible(lb1gt_1,288). happensAt(active(lb1gt_1),288). orientation(lb1gt_2,146,288). coords(lb1gt_2,186,152,288).  visible(lb1gt_2,288). happensAt(walking(lb1gt_2),288).
#show holdsAt/2.