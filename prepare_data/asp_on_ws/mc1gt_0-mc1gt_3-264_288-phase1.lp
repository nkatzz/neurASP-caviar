
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

	orientation(mc1gt_0,152,264). coords(mc1gt_0,182,175,264).  visible(mc1gt_0,264). happensAt(walking(mc1gt_0),264). orientation(mc1gt_3,132,264). coords(mc1gt_3,217,111,264).  visible(mc1gt_3,264). happensAt(walking(mc1gt_3),264).
orientation(mc1gt_0,152,265). coords(mc1gt_0,182,176,265).  visible(mc1gt_0,265). happensAt(walking(mc1gt_0),265). orientation(mc1gt_3,131,265). coords(mc1gt_3,217,114,265).  visible(mc1gt_3,265). happensAt(walking(mc1gt_3),265).
orientation(mc1gt_0,152,266). coords(mc1gt_0,182,177,266).  visible(mc1gt_0,266). happensAt(walking(mc1gt_0),266). orientation(mc1gt_3,131,266). coords(mc1gt_3,220,117,266).  visible(mc1gt_3,266). happensAt(walking(mc1gt_3),266).
orientation(mc1gt_0,152,267). coords(mc1gt_0,182,178,267).  visible(mc1gt_0,267). happensAt(walking(mc1gt_0),267). orientation(mc1gt_3,132,267). coords(mc1gt_3,223,119,267).  visible(mc1gt_3,267). happensAt(walking(mc1gt_3),267).
orientation(mc1gt_0,152,268). coords(mc1gt_0,182,179,268).  visible(mc1gt_0,268). happensAt(walking(mc1gt_0),268). orientation(mc1gt_3,132,268). coords(mc1gt_3,226,121,268).  visible(mc1gt_3,268). happensAt(walking(mc1gt_3),268).
orientation(mc1gt_0,152,269). coords(mc1gt_0,182,182,269).  visible(mc1gt_0,269). happensAt(walking(mc1gt_0),269). orientation(mc1gt_3,132,269). coords(mc1gt_3,228,123,269).  visible(mc1gt_3,269). happensAt(walking(mc1gt_3),269).
orientation(mc1gt_0,152,270). coords(mc1gt_0,183,185,270).  visible(mc1gt_0,270). happensAt(walking(mc1gt_0),270). orientation(mc1gt_3,132,270). coords(mc1gt_3,230,124,270).  visible(mc1gt_3,270). happensAt(walking(mc1gt_3),270).
orientation(mc1gt_0,152,271). coords(mc1gt_0,183,188,271).  visible(mc1gt_0,271). happensAt(walking(mc1gt_0),271). orientation(mc1gt_3,132,271). coords(mc1gt_3,231,125,271).  visible(mc1gt_3,271). happensAt(walking(mc1gt_3),271).
orientation(mc1gt_0,156,272). coords(mc1gt_0,184,191,272).  visible(mc1gt_0,272). happensAt(walking(mc1gt_0),272). orientation(mc1gt_3,132,272). coords(mc1gt_3,232,126,272).  visible(mc1gt_3,272). happensAt(walking(mc1gt_3),272).
orientation(mc1gt_0,156,273). coords(mc1gt_0,184,192,273).  visible(mc1gt_0,273). happensAt(walking(mc1gt_0),273). orientation(mc1gt_3,132,273). coords(mc1gt_3,233,126,273).  visible(mc1gt_3,273). happensAt(walking(mc1gt_3),273).
orientation(mc1gt_0,156,274). coords(mc1gt_0,185,194,274).  visible(mc1gt_0,274). happensAt(walking(mc1gt_0),274). orientation(mc1gt_3,135,274). coords(mc1gt_3,234,127,274).  visible(mc1gt_3,274). happensAt(walking(mc1gt_3),274).
orientation(mc1gt_0,156,275). coords(mc1gt_0,186,195,275).  visible(mc1gt_0,275). happensAt(walking(mc1gt_0),275). orientation(mc1gt_3,135,275). coords(mc1gt_3,234,128,275).  visible(mc1gt_3,275). happensAt(walking(mc1gt_3),275).
orientation(mc1gt_0,156,276). coords(mc1gt_0,186,195,276).  visible(mc1gt_0,276). happensAt(walking(mc1gt_0),276). orientation(mc1gt_3,135,276). coords(mc1gt_3,235,128,276).  visible(mc1gt_3,276). happensAt(walking(mc1gt_3),276).
orientation(mc1gt_0,156,277). coords(mc1gt_0,186,196,277).  visible(mc1gt_0,277). happensAt(walking(mc1gt_0),277). orientation(mc1gt_3,137,277). coords(mc1gt_3,237,129,277).  visible(mc1gt_3,277). happensAt(walking(mc1gt_3),277).
orientation(mc1gt_0,163,278). coords(mc1gt_0,186,196,278).  visible(mc1gt_0,278). happensAt(walking(mc1gt_0),278). orientation(mc1gt_3,135,278). coords(mc1gt_3,241,130,278).  visible(mc1gt_3,278). happensAt(walking(mc1gt_3),278).
orientation(mc1gt_0,163,279). coords(mc1gt_0,186,198,279).  visible(mc1gt_0,279). happensAt(walking(mc1gt_0),279). orientation(mc1gt_3,137,279). coords(mc1gt_3,245,130,279).  visible(mc1gt_3,279). happensAt(walking(mc1gt_3),279).
orientation(mc1gt_0,163,280). coords(mc1gt_0,186,199,280).  visible(mc1gt_0,280). happensAt(walking(mc1gt_0),280). orientation(mc1gt_3,137,280). coords(mc1gt_3,249,133,280).  visible(mc1gt_3,280). happensAt(walking(mc1gt_3),280).
orientation(mc1gt_0,163,281). coords(mc1gt_0,187,202,281).  visible(mc1gt_0,281). happensAt(walking(mc1gt_0),281). orientation(mc1gt_3,137,281). coords(mc1gt_3,252,136,281).  visible(mc1gt_3,281). happensAt(walking(mc1gt_3),281).
orientation(mc1gt_0,163,282). coords(mc1gt_0,189,204,282).  visible(mc1gt_0,282). happensAt(walking(mc1gt_0),282). orientation(mc1gt_3,136,282). coords(mc1gt_3,255,139,282).  visible(mc1gt_3,282). happensAt(walking(mc1gt_3),282).
orientation(mc1gt_0,163,283). coords(mc1gt_0,190,205,283).  visible(mc1gt_0,283). happensAt(walking(mc1gt_0),283). orientation(mc1gt_3,137,283). coords(mc1gt_3,257,140,283).  visible(mc1gt_3,283). happensAt(walking(mc1gt_3),283).
orientation(mc1gt_0,168,284). coords(mc1gt_0,190,207,284).  visible(mc1gt_0,284). happensAt(walking(mc1gt_0),284). orientation(mc1gt_3,135,284). coords(mc1gt_3,259,142,284).  visible(mc1gt_3,284). happensAt(walking(mc1gt_3),284).
orientation(mc1gt_0,168,285). coords(mc1gt_0,189,209,285).  visible(mc1gt_0,285). happensAt(walking(mc1gt_0),285). orientation(mc1gt_3,135,285). coords(mc1gt_3,260,143,285).  visible(mc1gt_3,285). happensAt(walking(mc1gt_3),285).
orientation(mc1gt_0,168,286). coords(mc1gt_0,189,209,286).  visible(mc1gt_0,286). happensAt(walking(mc1gt_0),286). orientation(mc1gt_3,135,286). coords(mc1gt_3,261,144,286).  visible(mc1gt_3,286). happensAt(walking(mc1gt_3),286).
orientation(mc1gt_0,168,287). coords(mc1gt_0,190,212,287).  visible(mc1gt_0,287). happensAt(walking(mc1gt_0),287). orientation(mc1gt_3,135,287). coords(mc1gt_3,262,145,287).  visible(mc1gt_3,287). happensAt(walking(mc1gt_3),287).
orientation(mc1gt_0,168,288). coords(mc1gt_0,191,213,288).  visible(mc1gt_0,288). happensAt(walking(mc1gt_0),288). orientation(mc1gt_3,135,288). coords(mc1gt_3,263,146,288).  visible(mc1gt_3,288). happensAt(walking(mc1gt_3),288).
#show holdsAt/2.