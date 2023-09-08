
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

	orientation(lb2gt_5,152,867). coords(lb2gt_5,80,102,867).  visible(lb2gt_5,867). happensAt(walking(lb2gt_5),867). orientation(lb2gt_6,151,867). coords(lb2gt_6,72,122,867).  visible(lb2gt_6,867). happensAt(inactive(lb2gt_6),867).
orientation(lb2gt_5,152,868). coords(lb2gt_5,78,102,868).  visible(lb2gt_5,868). happensAt(walking(lb2gt_5),868). orientation(lb2gt_6,151,868). coords(lb2gt_6,72,122,868).  visible(lb2gt_6,868). happensAt(inactive(lb2gt_6),868).
orientation(lb2gt_5,152,869). coords(lb2gt_5,76,102,869).  visible(lb2gt_5,869). happensAt(walking(lb2gt_5),869). orientation(lb2gt_6,151,869). coords(lb2gt_6,72,122,869).  visible(lb2gt_6,869). happensAt(inactive(lb2gt_6),869).
orientation(lb2gt_5,152,870). coords(lb2gt_5,76,103,870).  visible(lb2gt_5,870). happensAt(walking(lb2gt_5),870). orientation(lb2gt_6,151,870). coords(lb2gt_6,72,122,870).  visible(lb2gt_6,870). happensAt(inactive(lb2gt_6),870).
orientation(lb2gt_5,152,871). coords(lb2gt_5,75,104,871).  visible(lb2gt_5,871). happensAt(walking(lb2gt_5),871). orientation(lb2gt_6,151,871). coords(lb2gt_6,72,122,871).  visible(lb2gt_6,871). happensAt(inactive(lb2gt_6),871).
orientation(lb2gt_5,152,872). coords(lb2gt_5,75,105,872).  visible(lb2gt_5,872). happensAt(walking(lb2gt_5),872). orientation(lb2gt_6,151,872). coords(lb2gt_6,72,122,872).  visible(lb2gt_6,872). happensAt(inactive(lb2gt_6),872).
orientation(lb2gt_5,152,873). coords(lb2gt_5,75,106,873).  visible(lb2gt_5,873). happensAt(walking(lb2gt_5),873). orientation(lb2gt_6,151,873). coords(lb2gt_6,72,122,873).  visible(lb2gt_6,873). happensAt(inactive(lb2gt_6),873).
orientation(lb2gt_5,152,874). coords(lb2gt_5,75,108,874).  visible(lb2gt_5,874). happensAt(walking(lb2gt_5),874). orientation(lb2gt_6,151,874). coords(lb2gt_6,72,122,874).  visible(lb2gt_6,874). happensAt(inactive(lb2gt_6),874).
orientation(lb2gt_5,152,875). coords(lb2gt_5,75,108,875).  visible(lb2gt_5,875). happensAt(walking(lb2gt_5),875). orientation(lb2gt_6,151,875). coords(lb2gt_6,72,122,875).  visible(lb2gt_6,875). happensAt(inactive(lb2gt_6),875).
orientation(lb2gt_5,152,876). coords(lb2gt_5,75,109,876).  visible(lb2gt_5,876). happensAt(walking(lb2gt_5),876). orientation(lb2gt_6,151,876). coords(lb2gt_6,72,122,876).  visible(lb2gt_6,876). happensAt(inactive(lb2gt_6),876).
orientation(lb2gt_5,152,877). coords(lb2gt_5,74,110,877).  visible(lb2gt_5,877). happensAt(walking(lb2gt_5),877). orientation(lb2gt_6,151,877). coords(lb2gt_6,72,122,877).  visible(lb2gt_6,877). happensAt(inactive(lb2gt_6),877).
orientation(lb2gt_5,152,878). coords(lb2gt_5,74,111,878).  visible(lb2gt_5,878). happensAt(walking(lb2gt_5),878). orientation(lb2gt_6,151,878). coords(lb2gt_6,72,122,878).  visible(lb2gt_6,878). happensAt(inactive(lb2gt_6),878).
orientation(lb2gt_5,152,879). coords(lb2gt_5,73,111,879).  visible(lb2gt_5,879). happensAt(walking(lb2gt_5),879). orientation(lb2gt_6,151,879). coords(lb2gt_6,72,122,879).  visible(lb2gt_6,879). happensAt(inactive(lb2gt_6),879).
orientation(lb2gt_5,152,880). coords(lb2gt_5,72,111,880).  visible(lb2gt_5,880). happensAt(walking(lb2gt_5),880). orientation(lb2gt_6,151,880). coords(lb2gt_6,72,122,880).  visible(lb2gt_6,880). happensAt(inactive(lb2gt_6),880).
orientation(lb2gt_5,152,881). coords(lb2gt_5,72,111,881).  visible(lb2gt_5,881). happensAt(walking(lb2gt_5),881). orientation(lb2gt_6,151,881). coords(lb2gt_6,72,122,881).  visible(lb2gt_6,881). happensAt(inactive(lb2gt_6),881).
orientation(lb2gt_5,152,882). coords(lb2gt_5,72,112,882).  visible(lb2gt_5,882). happensAt(walking(lb2gt_5),882). orientation(lb2gt_6,151,882). coords(lb2gt_6,72,122,882).  visible(lb2gt_6,882). happensAt(inactive(lb2gt_6),882).
orientation(lb2gt_5,152,883). coords(lb2gt_5,72,112,883).  visible(lb2gt_5,883). happensAt(walking(lb2gt_5),883). orientation(lb2gt_6,151,883). coords(lb2gt_6,72,122,883).  visible(lb2gt_6,883). happensAt(inactive(lb2gt_6),883).
orientation(lb2gt_5,152,884). coords(lb2gt_5,72,112,884).  visible(lb2gt_5,884). happensAt(walking(lb2gt_5),884). orientation(lb2gt_6,151,884). coords(lb2gt_6,72,122,884).  visible(lb2gt_6,884). happensAt(inactive(lb2gt_6),884).
orientation(lb2gt_5,152,885). coords(lb2gt_5,72,112,885).  visible(lb2gt_5,885). happensAt(walking(lb2gt_5),885). orientation(lb2gt_6,151,885). coords(lb2gt_6,72,122,885).  visible(lb2gt_6,885). happensAt(inactive(lb2gt_6),885).
orientation(lb2gt_5,152,886). coords(lb2gt_5,72,112,886).  visible(lb2gt_5,886). happensAt(walking(lb2gt_5),886). orientation(lb2gt_6,151,886). coords(lb2gt_6,72,122,886).  visible(lb2gt_6,886). happensAt(inactive(lb2gt_6),886).
orientation(lb2gt_5,152,887). coords(lb2gt_5,72,112,887).  visible(lb2gt_5,887). happensAt(walking(lb2gt_5),887). orientation(lb2gt_6,151,887). coords(lb2gt_6,72,122,887).  visible(lb2gt_6,887). happensAt(inactive(lb2gt_6),887).
orientation(lb2gt_5,152,888). coords(lb2gt_5,72,112,888).  visible(lb2gt_5,888). happensAt(walking(lb2gt_5),888). orientation(lb2gt_6,151,888). coords(lb2gt_6,72,122,888).  visible(lb2gt_6,888). happensAt(inactive(lb2gt_6),888).
orientation(lb2gt_5,152,889). coords(lb2gt_5,72,112,889).  visible(lb2gt_5,889). happensAt(walking(lb2gt_5),889). orientation(lb2gt_6,151,889). coords(lb2gt_6,72,122,889).  visible(lb2gt_6,889). happensAt(inactive(lb2gt_6),889).
orientation(lb2gt_5,152,890). coords(lb2gt_5,72,112,890).  visible(lb2gt_5,890). happensAt(walking(lb2gt_5),890). orientation(lb2gt_6,151,890). coords(lb2gt_6,72,122,890).  visible(lb2gt_6,890). happensAt(inactive(lb2gt_6),890).
orientation(lb2gt_5,152,891). coords(lb2gt_5,72,112,891).  visible(lb2gt_5,891). happensAt(walking(lb2gt_5),891). orientation(lb2gt_6,151,891). coords(lb2gt_6,72,122,891).  visible(lb2gt_6,891). happensAt(inactive(lb2gt_6),891).
#show holdsAt/2.