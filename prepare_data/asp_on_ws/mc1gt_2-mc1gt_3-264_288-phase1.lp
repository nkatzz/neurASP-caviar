
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

	orientation(mc1gt_2,146,264). coords(mc1gt_2,156,151,264).  visible(mc1gt_2,264). happensAt(walking(mc1gt_2),264). orientation(mc1gt_3,132,264). coords(mc1gt_3,217,111,264).  visible(mc1gt_3,264). happensAt(walking(mc1gt_3),264).
orientation(mc1gt_2,148,265). coords(mc1gt_2,157,152,265).  visible(mc1gt_2,265). happensAt(walking(mc1gt_2),265). orientation(mc1gt_3,131,265). coords(mc1gt_3,217,114,265).  visible(mc1gt_3,265). happensAt(walking(mc1gt_3),265).
orientation(mc1gt_2,148,266). coords(mc1gt_2,157,153,266).  visible(mc1gt_2,266). happensAt(walking(mc1gt_2),266). orientation(mc1gt_3,131,266). coords(mc1gt_3,220,117,266).  visible(mc1gt_3,266). happensAt(walking(mc1gt_3),266).
orientation(mc1gt_2,149,267). coords(mc1gt_2,157,154,267).  visible(mc1gt_2,267). happensAt(walking(mc1gt_2),267). orientation(mc1gt_3,132,267). coords(mc1gt_3,223,119,267).  visible(mc1gt_3,267). happensAt(walking(mc1gt_3),267).
orientation(mc1gt_2,149,268). coords(mc1gt_2,158,155,268).  visible(mc1gt_2,268). happensAt(walking(mc1gt_2),268). orientation(mc1gt_3,132,268). coords(mc1gt_3,226,121,268).  visible(mc1gt_3,268). happensAt(walking(mc1gt_3),268).
orientation(mc1gt_2,152,269). coords(mc1gt_2,158,155,269).  visible(mc1gt_2,269). happensAt(walking(mc1gt_2),269). orientation(mc1gt_3,132,269). coords(mc1gt_3,228,123,269).  visible(mc1gt_3,269). happensAt(walking(mc1gt_3),269).
orientation(mc1gt_2,153,270). coords(mc1gt_2,160,156,270).  visible(mc1gt_2,270). happensAt(walking(mc1gt_2),270). orientation(mc1gt_3,132,270). coords(mc1gt_3,230,124,270).  visible(mc1gt_3,270). happensAt(walking(mc1gt_3),270).
orientation(mc1gt_2,154,271). coords(mc1gt_2,164,156,271).  visible(mc1gt_2,271). happensAt(walking(mc1gt_2),271). orientation(mc1gt_3,132,271). coords(mc1gt_3,231,125,271).  visible(mc1gt_3,271). happensAt(walking(mc1gt_3),271).
orientation(mc1gt_2,157,272). coords(mc1gt_2,166,157,272).  visible(mc1gt_2,272). happensAt(walking(mc1gt_2),272). orientation(mc1gt_3,132,272). coords(mc1gt_3,232,126,272).  visible(mc1gt_3,272). happensAt(walking(mc1gt_3),272).
orientation(mc1gt_2,153,273). coords(mc1gt_2,169,159,273).  visible(mc1gt_2,273). happensAt(walking(mc1gt_2),273). orientation(mc1gt_3,132,273). coords(mc1gt_3,233,126,273).  visible(mc1gt_3,273). happensAt(walking(mc1gt_3),273).
orientation(mc1gt_2,152,274). coords(mc1gt_2,171,159,274).  visible(mc1gt_2,274). happensAt(walking(mc1gt_2),274). orientation(mc1gt_3,135,274). coords(mc1gt_3,234,127,274).  visible(mc1gt_3,274). happensAt(walking(mc1gt_3),274).
orientation(mc1gt_2,152,275). coords(mc1gt_2,172,161,275).  visible(mc1gt_2,275). happensAt(walking(mc1gt_2),275). orientation(mc1gt_3,135,275). coords(mc1gt_3,234,128,275).  visible(mc1gt_3,275). happensAt(walking(mc1gt_3),275).
orientation(mc1gt_2,160,276). coords(mc1gt_2,174,162,276).  visible(mc1gt_2,276). happensAt(walking(mc1gt_2),276). orientation(mc1gt_3,135,276). coords(mc1gt_3,235,128,276).  visible(mc1gt_3,276). happensAt(walking(mc1gt_3),276).
orientation(mc1gt_2,161,277). coords(mc1gt_2,176,162,277).  visible(mc1gt_2,277). happensAt(walking(mc1gt_2),277). orientation(mc1gt_3,137,277). coords(mc1gt_3,237,129,277).  visible(mc1gt_3,277). happensAt(walking(mc1gt_3),277).
orientation(mc1gt_2,164,278). coords(mc1gt_2,176,163,278).  visible(mc1gt_2,278). happensAt(walking(mc1gt_2),278). orientation(mc1gt_3,135,278). coords(mc1gt_3,241,130,278).  visible(mc1gt_3,278). happensAt(walking(mc1gt_3),278).
orientation(mc1gt_2,164,279). coords(mc1gt_2,177,164,279).  visible(mc1gt_2,279). happensAt(walking(mc1gt_2),279). orientation(mc1gt_3,137,279). coords(mc1gt_3,245,130,279).  visible(mc1gt_3,279). happensAt(walking(mc1gt_3),279).
orientation(mc1gt_2,152,280). coords(mc1gt_2,177,164,280).  visible(mc1gt_2,280). happensAt(walking(mc1gt_2),280). orientation(mc1gt_3,137,280). coords(mc1gt_3,249,133,280).  visible(mc1gt_3,280). happensAt(walking(mc1gt_3),280).
orientation(mc1gt_2,152,281). coords(mc1gt_2,178,164,281).  visible(mc1gt_2,281). happensAt(walking(mc1gt_2),281). orientation(mc1gt_3,137,281). coords(mc1gt_3,252,136,281).  visible(mc1gt_3,281). happensAt(walking(mc1gt_3),281).
orientation(mc1gt_2,152,282). coords(mc1gt_2,178,165,282).  visible(mc1gt_2,282). happensAt(walking(mc1gt_2),282). orientation(mc1gt_3,136,282). coords(mc1gt_3,255,139,282).  visible(mc1gt_3,282). happensAt(walking(mc1gt_3),282).
orientation(mc1gt_2,152,283). coords(mc1gt_2,179,166,283).  visible(mc1gt_2,283). happensAt(walking(mc1gt_2),283). orientation(mc1gt_3,137,283). coords(mc1gt_3,257,140,283).  visible(mc1gt_3,283). happensAt(walking(mc1gt_3),283).
orientation(mc1gt_2,152,284). coords(mc1gt_2,180,168,284).  visible(mc1gt_2,284). happensAt(walking(mc1gt_2),284). orientation(mc1gt_3,135,284). coords(mc1gt_3,259,142,284).  visible(mc1gt_3,284). happensAt(walking(mc1gt_3),284).
orientation(mc1gt_2,152,285). coords(mc1gt_2,180,170,285).  visible(mc1gt_2,285). happensAt(walking(mc1gt_2),285). orientation(mc1gt_3,135,285). coords(mc1gt_3,260,143,285).  visible(mc1gt_3,285). happensAt(walking(mc1gt_3),285).
orientation(mc1gt_2,156,286). coords(mc1gt_2,183,173,286).  visible(mc1gt_2,286). happensAt(walking(mc1gt_2),286). orientation(mc1gt_3,135,286). coords(mc1gt_3,261,144,286).  visible(mc1gt_3,286). happensAt(walking(mc1gt_3),286).
orientation(mc1gt_2,165,287). coords(mc1gt_2,185,176,287).  visible(mc1gt_2,287). happensAt(walking(mc1gt_2),287). orientation(mc1gt_3,135,287). coords(mc1gt_3,262,145,287).  visible(mc1gt_3,287). happensAt(walking(mc1gt_3),287).
orientation(mc1gt_2,163,288). coords(mc1gt_2,187,178,288).  visible(mc1gt_2,288). happensAt(walking(mc1gt_2),288). orientation(mc1gt_3,135,288). coords(mc1gt_3,263,146,288).  visible(mc1gt_3,288). happensAt(walking(mc1gt_3),288).
#show holdsAt/2.