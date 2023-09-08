
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

	orientation(fcgt_0,150,12). coords(fcgt_0,149,119,12).  visible(fcgt_0,12). happensAt(walking(fcgt_0),12). orientation(fcgt_3,159,12). coords(fcgt_3,103,163,12).  visible(fcgt_3,12). happensAt(inactive(fcgt_3),12).
orientation(fcgt_0,150,13). coords(fcgt_0,150,119,13).  visible(fcgt_0,13). happensAt(walking(fcgt_0),13). orientation(fcgt_3,159,13). coords(fcgt_3,103,163,13).  visible(fcgt_3,13). happensAt(inactive(fcgt_3),13).
orientation(fcgt_0,150,14). coords(fcgt_0,151,121,14).  visible(fcgt_0,14). happensAt(walking(fcgt_0),14). orientation(fcgt_3,159,14). coords(fcgt_3,103,163,14).  visible(fcgt_3,14). happensAt(inactive(fcgt_3),14).
orientation(fcgt_0,150,15). coords(fcgt_0,152,122,15).  visible(fcgt_0,15). happensAt(walking(fcgt_0),15). orientation(fcgt_3,159,15). coords(fcgt_3,103,163,15).  visible(fcgt_3,15). happensAt(inactive(fcgt_3),15).
orientation(fcgt_0,150,16). coords(fcgt_0,155,122,16).  visible(fcgt_0,16). happensAt(walking(fcgt_0),16). orientation(fcgt_3,159,16). coords(fcgt_3,103,163,16).  visible(fcgt_3,16). happensAt(inactive(fcgt_3),16).
orientation(fcgt_0,150,17). coords(fcgt_0,158,123,17).  visible(fcgt_0,17). happensAt(walking(fcgt_0),17). orientation(fcgt_3,159,17). coords(fcgt_3,103,163,17).  visible(fcgt_3,17). happensAt(inactive(fcgt_3),17).
orientation(fcgt_0,150,18). coords(fcgt_0,159,123,18).  visible(fcgt_0,18). happensAt(walking(fcgt_0),18). orientation(fcgt_3,159,18). coords(fcgt_3,103,163,18).  visible(fcgt_3,18). happensAt(inactive(fcgt_3),18).
orientation(fcgt_0,150,19). coords(fcgt_0,161,123,19).  visible(fcgt_0,19). happensAt(walking(fcgt_0),19). orientation(fcgt_3,159,19). coords(fcgt_3,103,163,19).  visible(fcgt_3,19). happensAt(inactive(fcgt_3),19).
orientation(fcgt_0,150,20). coords(fcgt_0,163,123,20).  visible(fcgt_0,20). happensAt(walking(fcgt_0),20). orientation(fcgt_3,159,20). coords(fcgt_3,103,163,20).  visible(fcgt_3,20). happensAt(inactive(fcgt_3),20).
orientation(fcgt_0,150,21). coords(fcgt_0,164,123,21).  visible(fcgt_0,21). happensAt(walking(fcgt_0),21). orientation(fcgt_3,159,21). coords(fcgt_3,103,163,21).  visible(fcgt_3,21). happensAt(inactive(fcgt_3),21).
orientation(fcgt_0,150,22). coords(fcgt_0,164,123,22).  visible(fcgt_0,22). happensAt(walking(fcgt_0),22). orientation(fcgt_3,159,22). coords(fcgt_3,103,163,22).  visible(fcgt_3,22). happensAt(inactive(fcgt_3),22).
orientation(fcgt_0,150,23). coords(fcgt_0,165,123,23).  visible(fcgt_0,23). happensAt(walking(fcgt_0),23). orientation(fcgt_3,159,23). coords(fcgt_3,103,163,23).  visible(fcgt_3,23). happensAt(inactive(fcgt_3),23).
orientation(fcgt_0,150,24). coords(fcgt_0,165,123,24).  visible(fcgt_0,24). happensAt(walking(fcgt_0),24). orientation(fcgt_3,159,24). coords(fcgt_3,103,163,24).  visible(fcgt_3,24). happensAt(inactive(fcgt_3),24).
orientation(fcgt_0,150,25). coords(fcgt_0,166,123,25).  visible(fcgt_0,25). happensAt(walking(fcgt_0),25). orientation(fcgt_3,159,25). coords(fcgt_3,103,163,25).  visible(fcgt_3,25). happensAt(inactive(fcgt_3),25).
orientation(fcgt_0,150,26). coords(fcgt_0,166,124,26).  visible(fcgt_0,26). happensAt(walking(fcgt_0),26). orientation(fcgt_3,159,26). coords(fcgt_3,103,163,26).  visible(fcgt_3,26). happensAt(inactive(fcgt_3),26).
orientation(fcgt_0,150,27). coords(fcgt_0,166,124,27).  visible(fcgt_0,27). happensAt(walking(fcgt_0),27). orientation(fcgt_3,159,27). coords(fcgt_3,103,163,27).  visible(fcgt_3,27). happensAt(inactive(fcgt_3),27).
orientation(fcgt_0,150,28). coords(fcgt_0,169,124,28).  visible(fcgt_0,28). happensAt(walking(fcgt_0),28). orientation(fcgt_3,159,28). coords(fcgt_3,103,163,28).  visible(fcgt_3,28). happensAt(inactive(fcgt_3),28).
orientation(fcgt_0,150,29). coords(fcgt_0,171,125,29).  visible(fcgt_0,29). happensAt(walking(fcgt_0),29). orientation(fcgt_3,159,29). coords(fcgt_3,103,163,29).  visible(fcgt_3,29). happensAt(inactive(fcgt_3),29).
orientation(fcgt_0,150,30). coords(fcgt_0,174,126,30).  visible(fcgt_0,30). happensAt(walking(fcgt_0),30). orientation(fcgt_3,159,30). coords(fcgt_3,103,163,30).  visible(fcgt_3,30). happensAt(inactive(fcgt_3),30).
orientation(fcgt_0,150,31). coords(fcgt_0,175,127,31).  visible(fcgt_0,31). happensAt(walking(fcgt_0),31). orientation(fcgt_3,159,31). coords(fcgt_3,103,163,31).  visible(fcgt_3,31). happensAt(inactive(fcgt_3),31).
orientation(fcgt_0,150,32). coords(fcgt_0,175,129,32).  visible(fcgt_0,32). happensAt(walking(fcgt_0),32). orientation(fcgt_3,159,32). coords(fcgt_3,103,163,32).  visible(fcgt_3,32). happensAt(inactive(fcgt_3),32).
orientation(fcgt_0,150,33). coords(fcgt_0,177,130,33).  visible(fcgt_0,33). happensAt(walking(fcgt_0),33). orientation(fcgt_3,159,33). coords(fcgt_3,103,163,33).  visible(fcgt_3,33). happensAt(inactive(fcgt_3),33).
orientation(fcgt_0,150,34). coords(fcgt_0,177,131,34).  visible(fcgt_0,34). happensAt(walking(fcgt_0),34). orientation(fcgt_3,159,34). coords(fcgt_3,103,163,34).  visible(fcgt_3,34). happensAt(inactive(fcgt_3),34).
orientation(fcgt_0,150,35). coords(fcgt_0,177,132,35).  visible(fcgt_0,35). happensAt(walking(fcgt_0),35). orientation(fcgt_3,159,35). coords(fcgt_3,103,163,35).  visible(fcgt_3,35). happensAt(inactive(fcgt_3),35).
orientation(fcgt_0,150,36). coords(fcgt_0,178,133,36).  visible(fcgt_0,36). happensAt(walking(fcgt_0),36). orientation(fcgt_3,159,36). coords(fcgt_3,103,163,36).  visible(fcgt_3,36). happensAt(inactive(fcgt_3),36).
#show holdsAt/2.