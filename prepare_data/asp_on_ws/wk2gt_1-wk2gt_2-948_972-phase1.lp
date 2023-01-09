
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

	orientation(wk2gt_1,133,948). coords(wk2gt_1,121,48,948).  visible(wk2gt_1,948). happensAt(walking(wk2gt_1),948). orientation(wk2gt_2,118,948). coords(wk2gt_2,100,64,948).  visible(wk2gt_2,948). happensAt(active(wk2gt_2),948).
orientation(wk2gt_1,133,949). coords(wk2gt_1,121,48,949).  visible(wk2gt_1,949). happensAt(walking(wk2gt_1),949). orientation(wk2gt_2,118,949). coords(wk2gt_2,100,64,949).  visible(wk2gt_2,949). happensAt(active(wk2gt_2),949).
orientation(wk2gt_1,133,950). coords(wk2gt_1,121,48,950).  visible(wk2gt_1,950). happensAt(walking(wk2gt_1),950). orientation(wk2gt_2,118,950). coords(wk2gt_2,100,64,950).  visible(wk2gt_2,950). happensAt(active(wk2gt_2),950).
orientation(wk2gt_1,133,951). coords(wk2gt_1,122,47,951).  visible(wk2gt_1,951). happensAt(walking(wk2gt_1),951). orientation(wk2gt_2,118,951). coords(wk2gt_2,100,64,951).  visible(wk2gt_2,951). happensAt(active(wk2gt_2),951).
orientation(wk2gt_1,133,952). coords(wk2gt_1,122,47,952).  visible(wk2gt_1,952). happensAt(walking(wk2gt_1),952). orientation(wk2gt_2,118,952). coords(wk2gt_2,100,64,952).  visible(wk2gt_2,952). happensAt(active(wk2gt_2),952).
orientation(wk2gt_1,133,953). coords(wk2gt_1,123,48,953).  visible(wk2gt_1,953). happensAt(walking(wk2gt_1),953). orientation(wk2gt_2,118,953). coords(wk2gt_2,100,64,953).  visible(wk2gt_2,953). happensAt(active(wk2gt_2),953).
orientation(wk2gt_1,133,954). coords(wk2gt_1,123,48,954).  visible(wk2gt_1,954). happensAt(walking(wk2gt_1),954). orientation(wk2gt_2,118,954). coords(wk2gt_2,99,64,954).  visible(wk2gt_2,954). happensAt(active(wk2gt_2),954).
orientation(wk2gt_1,133,955). coords(wk2gt_1,123,47,955).  visible(wk2gt_1,955). happensAt(walking(wk2gt_1),955). orientation(wk2gt_2,122,955). coords(wk2gt_2,99,64,955).  visible(wk2gt_2,955). happensAt(active(wk2gt_2),955).
orientation(wk2gt_1,133,956). coords(wk2gt_1,123,46,956).  visible(wk2gt_1,956). happensAt(walking(wk2gt_1),956). orientation(wk2gt_2,122,956). coords(wk2gt_2,99,64,956).  visible(wk2gt_2,956). happensAt(active(wk2gt_2),956).
orientation(wk2gt_1,133,957). coords(wk2gt_1,123,46,957).  visible(wk2gt_1,957). happensAt(walking(wk2gt_1),957). orientation(wk2gt_2,126,957). coords(wk2gt_2,99,64,957).  visible(wk2gt_2,957). happensAt(active(wk2gt_2),957).
orientation(wk2gt_1,133,958). coords(wk2gt_1,124,46,958).  visible(wk2gt_1,958). happensAt(walking(wk2gt_1),958). orientation(wk2gt_2,126,958). coords(wk2gt_2,99,64,958).  visible(wk2gt_2,958). happensAt(active(wk2gt_2),958).
orientation(wk2gt_1,133,959). coords(wk2gt_1,124,46,959).  visible(wk2gt_1,959). happensAt(walking(wk2gt_1),959). orientation(wk2gt_2,128,959). coords(wk2gt_2,99,64,959).  visible(wk2gt_2,959). happensAt(active(wk2gt_2),959).
orientation(wk2gt_1,133,960). coords(wk2gt_1,125,45,960).  visible(wk2gt_1,960). happensAt(walking(wk2gt_1),960). orientation(wk2gt_2,128,960). coords(wk2gt_2,99,64,960).  visible(wk2gt_2,960). happensAt(active(wk2gt_2),960).
orientation(wk2gt_1,133,961). coords(wk2gt_1,125,45,961).  visible(wk2gt_1,961). happensAt(walking(wk2gt_1),961). orientation(wk2gt_2,128,961). coords(wk2gt_2,99,64,961).  visible(wk2gt_2,961). happensAt(active(wk2gt_2),961).
orientation(wk2gt_1,133,962). coords(wk2gt_1,125,45,962).  visible(wk2gt_1,962). happensAt(walking(wk2gt_1),962). orientation(wk2gt_2,132,962). coords(wk2gt_2,99,64,962).  visible(wk2gt_2,962). happensAt(active(wk2gt_2),962).
orientation(wk2gt_1,133,963). coords(wk2gt_1,125,45,963).  visible(wk2gt_1,963). happensAt(walking(wk2gt_1),963). orientation(wk2gt_2,137,963). coords(wk2gt_2,98,64,963).  visible(wk2gt_2,963). happensAt(active(wk2gt_2),963).
orientation(wk2gt_1,133,964). coords(wk2gt_1,125,45,964).  visible(wk2gt_1,964). happensAt(walking(wk2gt_1),964). orientation(wk2gt_2,137,964). coords(wk2gt_2,98,64,964).  visible(wk2gt_2,964). happensAt(active(wk2gt_2),964).
orientation(wk2gt_1,133,965). coords(wk2gt_1,126,45,965).  visible(wk2gt_1,965). happensAt(walking(wk2gt_1),965). orientation(wk2gt_2,135,965). coords(wk2gt_2,98,64,965).  visible(wk2gt_2,965). happensAt(active(wk2gt_2),965).
orientation(wk2gt_1,133,966). coords(wk2gt_1,127,45,966).  visible(wk2gt_1,966). happensAt(walking(wk2gt_1),966). orientation(wk2gt_2,135,966). coords(wk2gt_2,98,65,966).  visible(wk2gt_2,966). happensAt(active(wk2gt_2),966).
orientation(wk2gt_1,133,967). coords(wk2gt_1,127,45,967).  visible(wk2gt_1,967). happensAt(walking(wk2gt_1),967). orientation(wk2gt_2,135,967). coords(wk2gt_2,98,65,967).  visible(wk2gt_2,967). happensAt(active(wk2gt_2),967).
orientation(wk2gt_1,133,968). coords(wk2gt_1,127,45,968).  visible(wk2gt_1,968). happensAt(walking(wk2gt_1),968). orientation(wk2gt_2,135,968). coords(wk2gt_2,97,65,968).  visible(wk2gt_2,968). happensAt(active(wk2gt_2),968).
orientation(wk2gt_1,133,969). coords(wk2gt_1,127,45,969).  visible(wk2gt_1,969). happensAt(walking(wk2gt_1),969). orientation(wk2gt_2,137,969). coords(wk2gt_2,97,65,969).  visible(wk2gt_2,969). happensAt(active(wk2gt_2),969).
orientation(wk2gt_1,133,970). coords(wk2gt_1,128,44,970).  visible(wk2gt_1,970). happensAt(walking(wk2gt_1),970). orientation(wk2gt_2,137,970). coords(wk2gt_2,97,65,970).  visible(wk2gt_2,970). happensAt(active(wk2gt_2),970).
orientation(wk2gt_1,133,971). coords(wk2gt_1,128,44,971).  visible(wk2gt_1,971). happensAt(walking(wk2gt_1),971). orientation(wk2gt_2,137,971). coords(wk2gt_2,97,65,971).  visible(wk2gt_2,971). happensAt(active(wk2gt_2),971).
orientation(wk2gt_1,133,972). coords(wk2gt_1,128,44,972).  visible(wk2gt_1,972). happensAt(walking(wk2gt_1),972). orientation(wk2gt_2,137,972). coords(wk2gt_2,97,65,972).  visible(wk2gt_2,972). happensAt(active(wk2gt_2),972).
#show holdsAt/2.