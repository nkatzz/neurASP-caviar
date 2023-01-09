
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

	orientation(fcgt_0,150,12). coords(fcgt_0,149,119,12).  visible(fcgt_0,12). happensAt(walking(fcgt_0),12). orientation(fcgt_4,166,12). coords(fcgt_4,105,215,12).  visible(fcgt_4,12). happensAt(walking(fcgt_4),12).
orientation(fcgt_0,150,13). coords(fcgt_0,150,119,13).  visible(fcgt_0,13). happensAt(walking(fcgt_0),13). orientation(fcgt_4,166,13). coords(fcgt_4,105,215,13).  visible(fcgt_4,13). happensAt(walking(fcgt_4),13).
orientation(fcgt_0,150,14). coords(fcgt_0,151,121,14).  visible(fcgt_0,14). happensAt(walking(fcgt_0),14). orientation(fcgt_4,166,14). coords(fcgt_4,105,214,14).  visible(fcgt_4,14). happensAt(walking(fcgt_4),14).
orientation(fcgt_0,150,15). coords(fcgt_0,152,122,15).  visible(fcgt_0,15). happensAt(walking(fcgt_0),15). orientation(fcgt_4,166,15). coords(fcgt_4,105,214,15).  visible(fcgt_4,15). happensAt(walking(fcgt_4),15).
orientation(fcgt_0,150,16). coords(fcgt_0,155,122,16).  visible(fcgt_0,16). happensAt(walking(fcgt_0),16). orientation(fcgt_4,166,16). coords(fcgt_4,104,214,16).  visible(fcgt_4,16). happensAt(walking(fcgt_4),16).
orientation(fcgt_0,150,17). coords(fcgt_0,158,123,17).  visible(fcgt_0,17). happensAt(walking(fcgt_0),17). orientation(fcgt_4,166,17). coords(fcgt_4,104,214,17).  visible(fcgt_4,17). happensAt(walking(fcgt_4),17).
orientation(fcgt_0,150,18). coords(fcgt_0,159,123,18).  visible(fcgt_0,18). happensAt(walking(fcgt_0),18). orientation(fcgt_4,166,18). coords(fcgt_4,104,214,18).  visible(fcgt_4,18). happensAt(walking(fcgt_4),18).
orientation(fcgt_0,150,19). coords(fcgt_0,161,123,19).  visible(fcgt_0,19). happensAt(walking(fcgt_0),19). orientation(fcgt_4,166,19). coords(fcgt_4,104,214,19).  visible(fcgt_4,19). happensAt(walking(fcgt_4),19).
orientation(fcgt_0,150,20). coords(fcgt_0,163,123,20).  visible(fcgt_0,20). happensAt(walking(fcgt_0),20). orientation(fcgt_4,166,20). coords(fcgt_4,104,214,20).  visible(fcgt_4,20). happensAt(walking(fcgt_4),20).
orientation(fcgt_0,150,21). coords(fcgt_0,164,123,21).  visible(fcgt_0,21). happensAt(walking(fcgt_0),21). orientation(fcgt_4,166,21). coords(fcgt_4,104,214,21).  visible(fcgt_4,21). happensAt(walking(fcgt_4),21).
orientation(fcgt_0,150,22). coords(fcgt_0,164,123,22).  visible(fcgt_0,22). happensAt(walking(fcgt_0),22). orientation(fcgt_4,166,22). coords(fcgt_4,103,214,22).  visible(fcgt_4,22). happensAt(walking(fcgt_4),22).
orientation(fcgt_0,150,23). coords(fcgt_0,165,123,23).  visible(fcgt_0,23). happensAt(walking(fcgt_0),23). orientation(fcgt_4,166,23). coords(fcgt_4,103,214,23).  visible(fcgt_4,23). happensAt(walking(fcgt_4),23).
orientation(fcgt_0,150,24). coords(fcgt_0,165,123,24).  visible(fcgt_0,24). happensAt(walking(fcgt_0),24). orientation(fcgt_4,166,24). coords(fcgt_4,103,214,24).  visible(fcgt_4,24). happensAt(walking(fcgt_4),24).
orientation(fcgt_0,150,25). coords(fcgt_0,166,123,25).  visible(fcgt_0,25). happensAt(walking(fcgt_0),25). orientation(fcgt_4,166,25). coords(fcgt_4,103,214,25).  visible(fcgt_4,25). happensAt(walking(fcgt_4),25).
orientation(fcgt_0,150,26). coords(fcgt_0,166,124,26).  visible(fcgt_0,26). happensAt(walking(fcgt_0),26). orientation(fcgt_4,166,26). coords(fcgt_4,102,214,26).  visible(fcgt_4,26). happensAt(walking(fcgt_4),26).
orientation(fcgt_0,150,27). coords(fcgt_0,166,124,27).  visible(fcgt_0,27). happensAt(walking(fcgt_0),27). orientation(fcgt_4,166,27). coords(fcgt_4,101,214,27).  visible(fcgt_4,27). happensAt(walking(fcgt_4),27).
orientation(fcgt_0,150,28). coords(fcgt_0,169,124,28).  visible(fcgt_0,28). happensAt(walking(fcgt_0),28). orientation(fcgt_4,166,28). coords(fcgt_4,101,212,28).  visible(fcgt_4,28). happensAt(walking(fcgt_4),28).
orientation(fcgt_0,150,29). coords(fcgt_0,171,125,29).  visible(fcgt_0,29). happensAt(walking(fcgt_0),29). orientation(fcgt_4,166,29). coords(fcgt_4,101,212,29).  visible(fcgt_4,29). happensAt(walking(fcgt_4),29).
orientation(fcgt_0,150,30). coords(fcgt_0,174,126,30).  visible(fcgt_0,30). happensAt(walking(fcgt_0),30). orientation(fcgt_4,166,30). coords(fcgt_4,101,212,30).  visible(fcgt_4,30). happensAt(walking(fcgt_4),30).
orientation(fcgt_0,150,31). coords(fcgt_0,175,127,31).  visible(fcgt_0,31). happensAt(walking(fcgt_0),31). orientation(fcgt_4,166,31). coords(fcgt_4,101,212,31).  visible(fcgt_4,31). happensAt(walking(fcgt_4),31).
orientation(fcgt_0,150,32). coords(fcgt_0,175,129,32).  visible(fcgt_0,32). happensAt(walking(fcgt_0),32). orientation(fcgt_4,166,32). coords(fcgt_4,101,212,32).  visible(fcgt_4,32). happensAt(walking(fcgt_4),32).
orientation(fcgt_0,150,33). coords(fcgt_0,177,130,33).  visible(fcgt_0,33). happensAt(walking(fcgt_0),33). orientation(fcgt_4,166,33). coords(fcgt_4,101,212,33).  visible(fcgt_4,33). happensAt(walking(fcgt_4),33).
orientation(fcgt_0,150,34). coords(fcgt_0,177,131,34).  visible(fcgt_0,34). happensAt(walking(fcgt_0),34). orientation(fcgt_4,166,34). coords(fcgt_4,101,212,34).  visible(fcgt_4,34). happensAt(walking(fcgt_4),34).
orientation(fcgt_0,150,35). coords(fcgt_0,177,132,35).  visible(fcgt_0,35). happensAt(walking(fcgt_0),35). orientation(fcgt_4,166,35). coords(fcgt_4,101,212,35).  visible(fcgt_4,35). happensAt(walking(fcgt_4),35).
orientation(fcgt_0,150,36). coords(fcgt_0,178,133,36).  visible(fcgt_0,36). happensAt(walking(fcgt_0),36). orientation(fcgt_4,169,36). coords(fcgt_4,101,212,36).  visible(fcgt_4,36). happensAt(walking(fcgt_4),36).
#show holdsAt/2.